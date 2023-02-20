import xml.etree.ElementTree as ET

import numpy as np


class Anno_xml2list(object):
    """
    Attributes
    ----------
    classes : list
        list of VOC classes
    """

    def __init__(self, classes):
        self.classes = classes


    def __call__(self, xml_path, width, height):
        """
        Args:
            xml_path (str) : path for xml
            width (int) : image width
            height (int) : image height

        Returns:
            ret : [[xmin, ymin, xmax, ymax, label_index], ...]
        """

        # 画像内の全てのアノテーションのリスト
        ret = []

        # xml の解析
        xml = ET.parse(xml_path).getroot()

        for obj in xml.iter('object'):

            # アノテーションがdifficultは除外
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            # 各objectのアノテーションのリスト
            bndbox = []

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            # bbox
            pts = ['xmin', 'xmax', 'ymin', 'ymax']
            for pt in pts:
                cur_pixel = int(bbox.find(pt).text) - 1

                # 規格化
                if pt == 'xmin' or pt == 'xmax':
                    cur_pixel /= width
                else:
                    cur_pixel /= height

                bndbox.append(cur_pixel)
            # label
            label_id = self.classes.index(name)
            bndbox.append(label_id)

            ret += bndbox
        return np.array(ret)
