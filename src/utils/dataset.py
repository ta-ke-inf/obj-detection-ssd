import cv2
import numpy as np
import torch
import torch.utils.data as data


class VOCDataset(data.Dataset):
    """
    Attributes
    ----------
    img_list : list
        image file path

    anno_list : list
        annotetion file path

    phase : 'train' or 'val'

    transform : object
        preprocessing class instanse

    transform_anno : object
        xml to list transform class instanse
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno) -> None:
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
        self.transform_anno = transform_anno

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index) -> torch.Tensor:
        # 前処理後のTensorとアノテーション, H, W を取得
        img, gt, h, w = self._pull_item(index)
        return img, gt

    def _pull_item(self, index):
        # 画像の読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)
        height, width, channels = img.shape

        # xml to list
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # Datatransform
        img_transformed, boxes, labels = self.transform(img, self.phase, anno_list[:, :4], anno_list[:, 4]) # (H, W, C)
        # 画像の ndarray を Tensor に変換
        img_transformed = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1) # (C, W, H)
        # アノテーションの boxes と labels をつなげる
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img_transformed, gt, height, width
