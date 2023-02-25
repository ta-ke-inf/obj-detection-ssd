import torch.nn as nn
from DBox import DBox
from modules import make_extras, make_loc_conf, make_vgg


class SSD(nn.Module):
    def __init__(self, phase, cfg) -> None:
        super(SSD, self).__init__()

        self.phase = phase

        # SSD のネットワーク作成
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])

        # DBox の作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 推論時は Detect()
        if phase == "inference":
            self.detect = Detect()

if __name__ == "__main__":

    cfg = {
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

    ssd_test = SSD(phase="train", cfg=cfg)
    print(ssd_test)
