#検出結果となるBBoxを抽出する

import torch.nn as nn
from torch.autograd import Function


class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1) # conf のためのsoftmaxでクラスごとに正規化
        self.conf_thresh = conf_thresh # nm_supression の処理を軽くするため conf の値が 0.01 よりも大きい DBox のみ扱う
        self.top_k = top_k # nm_supression 内の top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        pass

if __name__ == "__main__":
    pass
