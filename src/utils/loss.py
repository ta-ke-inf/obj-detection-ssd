import torch.nn as nn


class MultiBoxLoss(nn.Module):
    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh # jaccard係数の閾値
        self.neg_pos = neg_pos # Hard Negative Mining の比率
        self.device = device # cpu or gpu

    def forward(self, predictions, targets):
        """
        Args:
            predictions (tuple): SSDの出力 (loc:[num_batch, 8732, 4]), (conf:[num_batch, 8732, 21]), (dbox_list:[8732, 4])
            targets (torch.Tensor): 正解 [num_batch, num_objs, 5]  [xmin, ymin, xmax, ymax, label_ind]

        Returns:
            loss_l: loc の損失
            loss_c: conf の損失
        """

        # predictions をばらす
        loc_data, conf_data, dbox_list = predictions

        # 要素数を把握
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)
