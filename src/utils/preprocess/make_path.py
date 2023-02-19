import os


def make_datapath_list(rootpath):
    """
    Args
    ---------
    rootpath : str
        path for data folder

    Returns
    ---------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        list for data path
    """

    # 画像とアノテーションのパスのテンプレート
    imgpath_templete = os.path.join(rootpath, 'JEGImages', '%s.png')
    annopath_templete = os.path.join(rootpath, 'Annotetions', '%s.xml')
    # train val 用のIDを取得
    train_id_names = os.path.join(rootpath, 'ImageSets/Main/train.txt')
    val_id_names = os.path.join(rootpath, 'ImageSets/Main/val.txt')

    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip() # 両端の不要なスペースと改行を削除
        train_img_list.append(imgpath_templete % file_id)
        train_anno_list.append(annopath_templete % file_id)

    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()
        val_img_list.append(imgpath_templete % file_id)
        val_anno_list.append(annopath_templete % file_id)

    return train_img_list, train_anno_list, val_img_list, val_anno_list
