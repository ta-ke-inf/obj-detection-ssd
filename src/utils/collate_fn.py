import torch


def od_collate_fn(batch):
    """
    Args:
        batch (list): ミニバッチの数分の画像のリスト

    Returns:
        imgs (Tensor): [mini_batch, 3, 300, 300]
        targets (list): アノテーション gt : Tensor[mini_batch, 5] のリスト
    """

    imgs = []
    targets = []

    for sample in batch:
        imgs.append(sample[0]) # sample[0] は img
        targets.append(torch.FloatTensor(sample[1])) # sample[1] は gt

    # list to Tensor
    imgs = torch.stack(imgs, dim=0) # 画像の結合 [mini_batch, 3, 300, 300]


    return imgs, targets
