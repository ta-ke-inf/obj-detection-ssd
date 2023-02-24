import torch
import torch.nn as nn
import torch.nn.init as init


class L2Norm(nn.Module):

    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        # torch.Tensor(input_channels): input_channels個のランダムな値
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale # 係数weight の初期値
        self.reset_parameters() # パラメータの初期化
        self.eps = 1e-10 # ゼロ除算を防ぐため


    def reset_parameter(self):
        # init.constant_: 定数で初期化
        init.constant_(self.weight, self.scale)


    def forward(self, x):

        # 正規化
        # pow(2): Tensor の二乗
        # .sum(dim=1, deepdim=True): [batch_num, 512, 35, 35] に対して 1次元目で次元を維持したまま合計 -> [batch_num, 1, 35, 35]
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # 各要素を norm で除算する
        x = torch.div(x, norm)

        # 係数 weight: self.weight は torch.Size([512]) なの unsqueeze で [1, 512, 1, 1] にして
        #       expand_as(x) で [batch_num, 512, 35, 35] にする
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out
