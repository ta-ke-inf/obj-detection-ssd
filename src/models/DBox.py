from itertools import product
from math import sqrt

import torch


class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()
        self.image_size = cfg['input_size']
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg['feature_maps'])
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.aspect_ratios = cfg['aspect_ratios']

    def make_dbox_list(self):
        mean = []

        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):

            # 特徴マップの各ピクセルに対し 4 or 6 個のDBoxの取得
            # 座標の組み合わせの取得
            for i, j in product(range(f), repeat=2):

                # 特徴量の画像サイズ
                # 300 / steps: [8, 16, 32, 64, 100, 300]
                f_k = self.image_size / self.steps[k]

                # DBox の中心座標 0~1 に規格化
                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k
                # 小さいDBox
                # 'min_sizes': [20, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # 大きいDBox
                # 'max_sizes': [60, 111, 162, 213, 264, 315]
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # 'aspect_ratios': [[2], [2,3], [2,3], [2,3], [2], [2]]
                for aspect in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(aspect), s_k/sqrt(aspect)]
                    mean += [cx, cy, s_k/sqrt(aspect), s_k*sqrt(aspect)]
        # list to torch.Tensor[8732*4]
        # view: torch.Tensor[8732*4] -> torch.Tensor[8734, 4]
        output = torch.Tensor(mean).view(-1, 4)
        # DBox のはみだしを防ぐため, 最大1, 最小0 にする
        output.clamp_(min=0, max=1)

        return output

if __name__ == "__main__":

    cfg = {
    # 入力画像のサイズ
    'input_size': 300,
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

    dbox = DBox(cfg)
    dbox_list = dbox.make_dbox_list()
    print(dbox_list.shape)
