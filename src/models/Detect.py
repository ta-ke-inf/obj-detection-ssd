#検出結果となるBBoxを抽出する

import torch
import torch.nn as nn
from box_utils import decode, nm_suppression
from torch.autograd import Function


class Detect(Function):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1) # conf のためのsoftmaxでクラスごとに正規化
        self.conf_thresh = conf_thresh # nm_supression の処理を軽くするため conf の値が 0.01 よりも大きい DBox のみ扱う
        self.top_k = top_k # nm_supression 内の top_k
        self.nms_thresh = nms_thresh

    def forward(self, loc_data, conf_data, dbox_list):
        """
        Args:
            loc_data (torch.Tensor): オフセット情報, [batch_num, 8732, 4]
            conf_data (torch.Tensor): 確信度, [batch_num, 8732, num_classes]
            dbox_list (torch.Tensor): DBox, [8732, 4]

        Returns:
            output (torch.Tensor): [batch_num, 21, 200, 5]
        """

        # 各サイズ
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)

        # conf を softmax でクラスごとに正規化
        conf_data = self.softmax(conf_data)

        # output の 初期化
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # conf_data: [batch_num, 8732, num_classes] -> conf_preds: [batch_num, num_classes ,8732]
        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):
            # loc と DBox から BBox を取得
            decoded_boxes = decode(loc_data[i], dbox_list) # [8732, 4]
            # conf のコピーを作成しておく
            conf_scores = conf_preds[i].clone() # [num_classes, 8732]

            # クラスごとに
            for cl in range(num_classes):
                # conf の値が 0.01 以上のみを処理する
                # .gt: greater than 0.01 -> 1, else -> 0
                c_mask = conf_scores[cl].gt(self.conf_thresh) # [8732]
                scores = conf_scores[cl][c_mask] # [閾値を超えたBBoxの数]

                # 0.01 を超えるものがない場合つまり scoresの要素数が0の場合はスキップ
                if scores.nelement() == 0:
                    continue


if __name__ == "__main__":
    pass
