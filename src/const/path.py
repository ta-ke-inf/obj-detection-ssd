from os.path import abspath, dirname, join

_PROJECT_ROOT = join(abspath(dirname(__file__)), "../..")

SOURCE_PATH = join(_PROJECT_ROOT, "src")
SAVE_PATH = join(_PROJECT_ROOT, "src/save")
DATA_PATH = join(_PROJECT_ROOT, "pytorch_advanced/2_objectdetection/data/VOCdevkit/VOC2012/")
UTILES_PATH = join(_PROJECT_ROOT, "pytorch_advanced/2_objectdetection/utils")
