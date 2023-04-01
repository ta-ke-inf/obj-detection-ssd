import os
import random
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from const.path import DATA_PATH, SOURCE_PATH, UTILES_PATH
from models.SSD import SSD
from utils.collate_fn import od_collate_fn
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

    #============================#
    #           Dataset          #
    #============================#

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

    #============================#
    #        Dataloader          #
    #============================#

    batch_size = 16

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


    dataloader_dict = {"train": train_loader, "val": val_loader}

    ssd_cfg = {
    # クラスの数
    'num_classes': 21,
    # 入力画像のサイズ
    'input_size': 300,
    # source 毎の出力する BBox の数
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],
    # 特徴量の画像サイズ
    'feature_maps': [38, 19, 10, 5, 3, 1],
    # DBox のサイズ
    'steps': [8, 16, 32, 64, 100, 300],
    # 小さいDBox のサイズ
    'min_sizes': [20, 60, 111, 162, 213, 264],
    # 大きいDBox のサイズ
    'max_sizes': [60, 111, 162, 213, 264, 315],
    # アスペクト比
    'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2], [2]]
    }

    net = SSD(phase="train", cfg=ssd_cfg)
    print(net)
