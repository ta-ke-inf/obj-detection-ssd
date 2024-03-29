import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from const.path import DATA_PATH, SAVE_PATH
from models.SSD import SSD
from utils.collate_fn import od_collate_fn
from utils.dataset import VOCDataset
from utils.loss import MultiBoxLoss
from utils.preprocess.DataTransform import DataTransform
from utils.preprocess.make_path import make_datapath_list
from utils.preprocess.xml_to_list import Anno_xml2list


# initialize He weights
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class Trainer:

    def __init__(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ) -> None:
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device


    def train_step(
            self,
            images: torch.Tensor,
            targets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            images (torch.Tensor): images each epoch, [num_batch, 3, 300, 300]
            targets (List[torch.Tensor]): targets each epoch, List[num_batch, torch.Size([num_objs, 5])]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [loss, output]
        """
        self.net.train()
        outputs = self.net(images)
        loss_l, loss_c = self.criterion(outputs, targets)
        loss = loss_l + loss_c

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), clip_value=2.0)
        self.optimizer.step()

        return loss, outputs


    def val_step(
            self,
            images: torch.Tensor,
            targets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images (torch.Tensor): images each epoch, [num_batch, 3, 300, 300]
            targets (List[torch.Tensor]): targets each epoch, List[num_batch, torch.Size([num_objs, 5])]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: [loss, output]
        """
        self.net.train()
        outputs = self.net(images)
        loss_l, loss_c = self.criterion(outputs, targets)
        loss = loss_l + loss_c

        return loss, outputs

    def fit(
            self, train_loader: data.DataLoader, val_loader: data.DataLoader, iteration: int
    ) -> Tuple[List[float], List[float], int]:
        """
        Args:
            train_loader (data.DataLoader): train dataloader
            val_loader (data.DataLoader): val dataloader
            iteration (int): start iteration

        Returns:
            Tuple[List[float], List[float], int]: [train loss per epoch, val loss per epoch, end iteration]
        """
        # train
        train_losses: List[float] = []

        for images, targets in train_loader:
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            loss, _ = self.train_step(images, targets)

            #images = images.to("cpu")
            #targets = targets.to("cpu")

            if iteration % 10 ==0:
                print(f"{iteration} iteration, train loss: {loss.item()}")
            train_losses.append(loss.item())
            iteration += 1

        # val
        val_losses: List[float] = []

        for images, targets in val_loader:
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]
            loss, _ = self.val_step(images, targets)

            #images = images.to("cpu")
            #targets = targets.to("cpu")
            val_losses.append(loss.item())


        return train_losses, val_losses, iteration


if __name__ == "__main__":

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(DATA_PATH)

    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

    color_mean = (104, 117, 123) # BGR mean
    input_size = 300 # 前処理後を300の正方形にリサイズ

    """
    define Dataset
    """
    train_dataset = VOCDataset(
        train_img_list,
        train_anno_list,
        phase='train',
        transform=DataTransform(input_size, color_mean),
        transform_anno=Anno_xml2list(voc_classes)
        )

    val_dataset = VOCDataset(
        val_img_list,
        val_anno_list,
        phase='val',
        transform=DataTransform(input_size, color_mean),
        transform_anno=Anno_xml2list(voc_classes)
    )
    """
    define Dataloader
    """
    batch_size = 32

    train_loader = data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        collate_fn=od_collate_fn
        )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        collate_fn=od_collate_fn
        )


    """
    define Network
    """
    ssd_cfg = {
    'num_classes': 21,
    'input_size': 300,
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [20, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2], [2]]
    }
    net = SSD(phase="train", cfg=ssd_cfg)

    # initialize weights
    vgg_weights = torch.load('./weights/vgg16_reducedfc.pth')
    net.vgg.load_state_dict(vgg_weights)

    net.extras.apply(weight_init)
    net.loc.apply(weight_init)
    net.conf.apply(weight_init)

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print("Your device: ", device)

    # define loss function
    criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)
    # define optimizer
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    """
    train or val phaze
    """
    net.to(device)
    torch.backends.cudnn.benchmark = True
    trainer = Trainer(net, optimizer, criterion, device)

    train_losses :List[float] = []
    val_losses :List[float] = []
    iteration = 1
    num_epochs = 10
    for i in range(num_epochs + 1):
        print(f"Epoch {i+1}/{num_epochs} {'-'*20} \n")

        train_losses_per_epoch, val_losses_per_epoch, iteration = trainer.fit(train_loader, val_loader, iteration)

        epoch_train_loss = sum(train_losses_per_epoch)
        epoch_val_loss = sum(val_losses_per_epoch)

        train_losses.append(train_losses_per_epoch)
        val_losses.append(val_losses_per_epoch)

        print(f"Epoch train loss: {epoch_train_loss}")
        print(f"Epoch val loss: {epoch_val_loss} \n")

        if((i+1) % 10 ==0):
            torch.save(trainer.net, os.path.join(SAVE_PATH, f"epoch_{i+1}.pt"))
