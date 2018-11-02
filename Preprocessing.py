import cv2, os
from align import AlignDlib
import numpy as np

from urllib.request import urlopen


def download_landmarks(dst_file):
    url = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
    decompressor = bz2.BZ2Decompressor()

    with urlopen(url) as src, open(dst_file, 'wb') as dst:
        data = src.read(1024)
        while len(data) > 0:
            dst.write(decompressor.decompress(data))
            data = src.read(1024)


dst_dir = 'models'
dst_file = os.path.join(dst_dir, 'landmarks.dat')

if not os.path.exists(dst_file):
    os.makedirs(dst_dir)
    download_landmarks(dst_file)


alignment = AlignDlib('/Users/siddhartham/Sid/PycharmProjects/Deep-Face-Recognition-Using-Inception-Model-Keras-OpenCV-Dlib/models/landmarks.dat')

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)


def convertToGrayscale(img_frame):
    return cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

def convertToColor(img_frame):
    return cv2.cvtColor(img_frame, cv2.COLOR_GRAY2BGR)

def load_image(path):
    img = cv2.imread(path, 1)
    return img[..., ::-1]


class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)


def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        num = 0
        if ".DS_Store" in i:
            continue
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            if num > 100:
                break
            num += 1
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)