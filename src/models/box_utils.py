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
        keep (torch.Tensor):
    """

    # return のひな型の作成
    count = 0
    # keep: torch.Size([確信度閾値(0.01)を超えたBBoxの数]). 要素は全て 0
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



if __name__ == "__main__":

    pass
