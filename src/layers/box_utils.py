import torch


def decode(loc, dbox_list) -> torch.Tensor:
    """
    Args:
        loc (ModuleList): SSD モデルで推論するオフセット情報 [8732, 4], [Δcx, Δcy, Δwidth, Δheight]
        dbox_list (ModuleList): DBox [8732, 4], [cx, cy, width, height]

    Returns:
        boxes (torch.Tensor([8732, 4])): BBox [8732, 4], [xmin, xmin, ymin, ymax]
    """

    bbox_cx_cy = dbox_list[:, :2] + 0.1*loc[:, :2] * dbox_list[:, 2:]
    bbox_w_h = dbox_list[:, 2:] * torch.exp(0.2*loc[:, 2:])
    boxes = torch.cat((bbox_cx_cy, bbox_w_h), dim=1)

    # [cx, cy, width, height] to [xmin, ymin, xmax, ymax]
    boxes[:, :2] -= boxes[:, 2:] / 2 # (xmin, ymin)
    boxes[:, 2:] += boxes[:, :2] # (xmax, ymax)

    return boxes


def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Args:
        boxes (torch.Tensor): [確信度閾値(0.01)を超えたBBoxの数, 4] -> BBox情報
        scores (torch.Tensor): [確信度閾値(0.01)を超えたBBoxの数] -> confの情報
        overlap (float, optional): BBoxの被り度合いの閾値. Defaults to 0.45.
        top_k (int, optional): _description_. Defaults to 200.

    Returns:
        keep (torch.Tensor): nms を通過した BBox の idx
        count (int): nms を通過した BBox の数
    """

    # return のひな型の作成
    count = 0
    # keep: torch.Size([確信度閾値(0.01)を超えたBBoxの数]). 要素は全て 0
    # nms の出力を格納する
    keep = scores.new(scores.size(0)).zero_().long()

    # 各 BBox の面積 area の計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # torch.mul: 要素同士の積
    area = torch.mul(x2 - x1, y2 - y1)

    # IOUの計算の際にひな形として使用
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    # scores を昇順に並び替える
    v, idx = scores.sort(0)

    # 後ろから top_k=200 個のBBox の index を取得
    idx = idx[-top_k:]

    # numel: tensor の要素数. 200
    while idx.numel() > 0:
        # conf が最大の index を取得する
        i = idx[-1]
        # conf が最大の index を格納
        keep[count] = i
        count += 1
        # 最大のconfを格納したので index を一つ減らす
        idx = idx[:-1]

        # keep に格納した 最大のconf のBBoxとそれ以外の idx のBBox の重なり度合いを計算指定していく
        # 最大の conf の BBox 以外の BBox を取得
        torch.index_select(x1, 0, idx, out=tmp_x1) # x1 のうち idx で指定された BBox のみ out に出力される
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

        # 現在の最大の conf (i) と それ以外(idx)の 重なった矩形領域を clamp で計算
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        # w, h の形も idx に合わせる
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

        # 重なった矩形領域の面積を計算
        inter = tmp_w * tmp_h

        # IoU の計算

        # idx の面積
        rem_area = torch.index_select(area, 0, idx)
        # 二つのエリアの和(OR)
        union = (rem_area - inter) + area[i]
        IoU = inter / union # IoU は idx の数分だけの数ある

        # IoU が overlap より小さい idx のみ残す
        # つまり、同じ物体だと思われるものは統合する
        idx = idx[IoU.le(overlap)]

    return keep, count


if __name__ == "__main__":

    pass
