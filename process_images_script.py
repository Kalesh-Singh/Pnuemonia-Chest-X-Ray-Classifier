# Required modules and imports
from PIL import Image
from glob import glob
import os


# Expected file structure for raw data

# . |-data |-chest_xray_raw |-test  |- NORMAL
#                                   |- PNUEMONIA
#                           |-train |- NORMAL
#                                   |- PNUEMONIA
#                           |-val   |- NORMAL
#                                   |- PNUEMONIA


# Store processed data in the following file structure

# . |-data |-chest_xray_processed |-test  |- NORMAL
#                                         |- PNUEMONIA
#                                 |-train |- NORMAL
#                                         |- PNUEMONIA
#                                 |-val   |- NORMAL
#                                         |- PNUEMONIA


# Constants
RAW_IMG_DIRS = [
    './data/chest_xray_raw/test/NORMAL',
    './data/chest_xray_raw/test/PNEUMONIA',
    './data/chest_xray_raw/train/NORMAL',
    './data/chest_xray_raw/train/PNEUMONIA',
    './data/chest_xray_raw/val/NORMAL',
    './data/chest_xray_raw/val/PNEUMONIA'
]

PROCESSED_IMG_DIR = 'chest_xray_processed'

IMG_WIDTH = 244
IMG_HEIGHT = 244
IMG_EXT = 'jpeg'

START_MSG = 'Processing images ...\nThis may take up to several minutes ...'
SUCCESS_MSG = 'Sucessfully resized and saved new images in ./data/' + PROCESSED_IMG_DIR + '/' + '\nDone.'


def getProcessedPath(rawPath):
    """Returns the path to save the resized image."""
    rawPath = rawPath.replace('\\', '/')
    dirs = rawPath.split('/')
    dirs[2]= PROCESSED_IMG_DIR
    new_dir = '/'.join(dirs[:-1])
    if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    return '/'.join(dirs)


def resizeAndSaveImg(path):
    """Resizes and saves the image."""
    img = Image.open(path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
    img.save(getProcessedPath(path))


def resizeAllImages():
    """Resizes all the raw images."""
    print(START_MSG)
    for _dir in RAW_IMG_DIRS:
        img_paths = glob(_dir + '/*.' + IMG_EXT)
        for path in img_paths:
            resizeAndSaveImg(path)
    print(SUCCESS_MSG)


resizeAllImages()
