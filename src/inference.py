from models.SSD import SSD
import torch
import cv2
import matplotlib.pyplot as plt
from utils.preprocess.DataTransform import DataTransform

def read_img(img_file_path) -> tuple:
    """
    Args:
        img_file_path (str): image file path
    Returns:
        tuple: (img, height, width, channels)
    """
    img = cv2.imread(img_file_path)
    height, width, channels = img.shape
    return img, height, width, channels

def show_img(img) -> None:
    """
    Args:
        img (str): opencv image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

def inference(img):
    """
    Args:
        img (str): opencv image [height, width, channels]
    Returns:
        detections (torch.Tensor): [num_batch, num_classes, num_detections, 5]
    """
    net.eval()
    x = img.unsqueeze(0) # [1, 3, 300, 300]
    detections = net(x)
    return detections


if __name__ == "__main__":
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

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

    net = SSD(phase="inference", cfg=ssd_cfg)

    net_weights = torch.load("../pytorch_advanced/2_objectdetection/weights/ssd300_mAP_77.43_v2.pth", map_location={'cuda:0': 'cpu'})
    net.load_state_dict(net_weights)

    img, height, width, channels = read_img("../pytorch_advanced/2_objectdetection/data/cowboy-757575_640.jpg")
    #show_img(img)

    color_mean = (104, 117, 123) # BGR mean
    input_size = 300 # resize to 300x300
    transform = DataTransform(input_size, color_mean)
    phase = "val"
    img_transformed, boxes, labels = transform(img, phase, "", "")
    img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

    detections = inference(img)
    print(detections.shape)
