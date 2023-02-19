import os
import random
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from const.path import DATA_PATH
from utils.preprocess.make_path import make_datapath_list

if __name__ == "__main__":

    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(DATA_PATH)
