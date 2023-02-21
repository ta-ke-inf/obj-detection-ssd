import os
import random
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from const.path import DATA_PATH, SOURCE_PATH, UTILES_PATH
from utils.dataset import VOCDataset
from utils.preprocess.DataTransform import DataTransform
from utils.preprocess.make_path import make_datapath_list
from utils.preprocess.xml_to_list import Anno_xml2list

if __name__ == "__main__":

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(DATA_PATH)

    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']

    color_mean = (104, 117, 123) # BGR mean
    input_size = 300 # 前処理後を300の正方形にリサイズ

    #=========================#
    #        Dataset          #
    #=========================#

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

    print(val_dataset.__getitem__(0))
