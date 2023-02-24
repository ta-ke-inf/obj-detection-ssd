from itertools import product


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

        # 'feature_maps': [38, 19, 10, 5, 1]
        for k, f in enumerate(self.feature_maps):
            # 座標の組み合わせの取得
            for i, j in product(range(f), repeat=2):
                pass
