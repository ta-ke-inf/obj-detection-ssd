import torch
import torch.nn as nn
import torch.nn.functional as F


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

        # 各 DBox に対して一番近い BBox のラベルを格納 ( background を 0 )
        conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
        # 各 DBox に対して一番近い BBox の補正情報を格納
        loc_t = torch.LongTensor(num_batch, num_dbox, 4).to(self.device)

        for idx in range(num_batch):
            # 正解 BBox の位置情報
            truths = targets[idx][:, -1].to(self.device)
            # 正解 BBox のラベル
            labels = targets[idx][:, :-1].to(self.device)

            variance = [0.1, 0.2]
            # loc_t と conf_t_label が上書きされる
            # loc_t : [num_batch, 8732, 4]
            # conf_t_label : [num_batch, 8732]
            match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)

        #=============
        # loss_l の計算
        #=============

        # Positive DBox のマスクを取得
        pos_mask = conf_t_label > 0 # [num_batch, 8732]
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data) # mask を適用させるために [num_batch, 8732] -> [num_batch, 8732, 4] に変形

        # mask を適用すると 1次元に圧縮される仕様なので, view で 次元を増やしている
        loc_t = loc_t[pos_idx].view(-1, 4) # 教師データ(正解BBoxに対するオフセット) : [物体を検出したBBoxの数(全ミニバッチの合計), 4]
        loc_p = loc_data[pos_idx].view(-1, 4) # 予測データ(予測BBoxに対するオフセット) : [物体を検出したBBoxの数(全ミニバッチの合計), 4]

        # Positive DBox に対する損失を計算
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')


        #=============
        # loss_c の計算
        #=============

        # [num_batch, 8732, 21] -> batch_conf: [num_batch*8732, 21]
        batch_conf = conf_data.view(-1, num_classes)
        # 各ミニバッチの8732個の各DBoxに対して損失を計算
        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction='none') # loss_c: [num_batch*8732]

        # ここから Hard Negative Minig --------------------------------------------

        # Positive DBox の数
        # num_mask: [num_batch, 1]
        num_pos = pos_mask.long().sum(1, keepdim=True) # .long(): Bool to int64
        loss_c = loss_c.view(num_batch, -1) # [num_batch*8732] -> [num_batch, 8732]

        loss_c = loss_c[pos_mask]
