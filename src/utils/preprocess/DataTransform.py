from .data_augumentation import (Compose, ConvertFromInts, Expand,
                                 PhotometricDistort, RandomMirror,
                                 RandomSampleCrop, Resize, SubtractMeans,
                                 ToAbsoluteCoords, ToPercentCoords)


class DataTransform():
    """
    Attributes
    ----------
    input_size : int
        input image size

    color_mean : (B, G, R)
        mean of each color channel
    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(), # int to float32
                ToAbsoluteCoords(), # アノテーションの規格化を戻す
                PhotometricDistort(), # 画像の色調をランダムに変化
                Expand(color_mean), # 画像のキャンパスを広げる
                RandomSampleCrop(), # 画像内の部分をランダムに抜き出す
                RandomMirror(), # 画像を反転
                ToPercentCoords(), # アノテーションの0~1規格化
                Resize(input_size), # 画像サイズを input_size の正方形にリサイズ
                SubtractMeans(color_mean) # BGR の平均値を引き算
            ]),

            'val': Compose([
                ConvertFromInts(), # int to float32
                Resize(input_size), # 画像サイズを input_size の正方形にリサイズ
                SubtractMeans(color_mean) # BGR の平均値を引き算
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        """
        Prameters
        ---------
        phase: 'train' or 'val'
        """

        return self.data_transform[phase](img, boxes, labels)
