from models.SSD import SSD
import torch

def test_ssd() -> None:

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

    # input images
    num_batch: int = 32
    channels: int = 3
    height: int = 300
    width: int = 300

    images = torch.randn(num_batch, channels, height, width)
    output = net(images)

    # params
    num_dbox = 8732
    num_classes = 21

    # loc
    assert (
        output[0].shape,
        output[1].shape,
        output[2].shape
        ) == (
        (num_batch, num_dbox, 4),
        (num_batch, num_dbox, num_classes),
        (num_dbox, 4)
        )
