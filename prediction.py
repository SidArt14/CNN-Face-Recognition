import cv2, random
import os, numpy as np
from PIL import Image

from align import AlignDlib
from Preprocessing import convertToGrayscale, align_image, load_metadata, load_image
from Classification import Classifier

path = "/Users/siddhartham/Sid/PycharmProjects/Deep-Face-Recognition-Using-Inception-Model-Keras-OpenCV-Dlib/FaceRecognition_dataset"

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

metadata = load_metadata(path)

classifier = Classifier(method='svc')
classifier.trainClassifier(metadata)

from glob import glob
#img_mask = path + '/**/' + '*.jpg'
user_names = glob(path+"/*")


for user in user_names:
    images = glob(user + "/*.jpg")

    index = random.randrange(len(images))

    img_path = images[index]

    print(img_path)
    img = load_image(img_path)

    img_class = classifier.classify(img)
    print(img_class)


test_path = "/Users/siddhartham/Sid/PycharmProjects/Deep-Face-Recognition-Using-Inception-Model-Keras-OpenCV-Dlib/Edmund_photos/Edmund_id_2.png"

#image = Image.open(test_path)
#print(type(image))
#image_data = np.asarray(image)
#print(type(image_data))

img = load_image(test_path)

img_class = classifier.classify(img)


cv2.imshow('im',img_class)
print("Image class is ",img_class)
