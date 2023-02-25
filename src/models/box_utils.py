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
