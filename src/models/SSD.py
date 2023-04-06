import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.DBox import DBox
from layers.Detect import Detect
from layers.L2Norm import L2Norm
from layers.modules import make_extras, make_loc_conf, make_vgg


class SSD(nn.Module):
    def __init__(self, phase, cfg) -> None:
        super(SSD, self).__init__()

        self.phase = phase
        self.num_classes = cfg["num_classes"]

        # SSD のネットワーク作成
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])

        # DBox の作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        # 推論時は Detect()
        if phase == "inference":
            self.detect = Detect()

    def forward(self, x):
        sources = list() # source1~6の出力を格納
        loc = list() # loc の出力を格納
        conf = list() # conf の出力を格納

        for k in range(23):
            x = self.vgg[k](x)
        source1 = self.L2Norm(x) # source1
        sources.append(source1)

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x) # source2

        # extras から source 3~6 を取得
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # l(x), c(x) は [N,C=アスペクトの数 * 4,H,W] -> [N,H,W,C=アスペクトの数 * 4]
        # これはアスペクト比の種類で異なるのが面倒なので
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # torch.cat([[N, N以降の要素数], [N, N以降の要素数], ...])
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1) # [batch_num, 8732*4]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1) # [batch_num, 8732*21]

        loc = loc.view(loc.size(0), -1, 4) # [batch_num, 8732, 4]
        conf = conf.view(conf.size(0), -1, self.num_classes) # [batch_num, 8732, 21]

        # まとめる
        output = [loc, conf, self.dbox_list]

        if self.phase == "inference":
            return self.detect(output[0], output[1], output[2])

        else:
            return output
