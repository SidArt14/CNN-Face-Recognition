
import csv
from glob import glob
import cv2
import datetime, time
from align import AlignDlib
import bz2
import pickle, timeit, random
import logging
import sys
import numpy as np
import os.path
import cv2, random
import os, numpy as np
import dlib

from align import AlignDlib
from Preprocessing import convertToGrayscale, align_image, load_metadata, load_image, convertToColor
from Classification import Classifier



from urllib.request import urlopen


def transformations(src, choice):
    if choice == 0:
        # Rotate 90
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
    if choice == 1:
        # Rotate 90 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_CLOCKWISE)
        src = cv2.flip(src, flipCode=1)
    if choice == 2:
        # Rotate 180
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
    if choice == 3:
        # Rotate 180 and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_180)
        src = cv2.flip(src, flipCode=1)
    if choice == 4:
        # Rotate 90 counter-clockwise
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    if choice == 5:
        # Rotate 90 counter-clockwise and flip horizontally
        src = cv2.rotate(src, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        src = cv2.flip(src, flipCode=1)

    return src




#cascadePath = "/Users/siddhartham/Sid/PycharmProjects/CNN-face-recognition/haarcascade_profileface.xml"
#cascadePath = "/Users/siddhartham/Sid/PycharmProjects/CNN-face-recognition/haarcascade_frontalface_default.xml"
cascadePath = dlib.get_frontal_face_detector()
#cascadePath = "HS.xml"
#faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX



#path = "/Users/siddhartham/Sid/PycharmProjects/Face-Recognition/dataset"
path = "/Users/siddhartham/Sid/PycharmProjects/Deep-Face-Recognition-Using-Inception-Model-Keras-OpenCV-Dlib/FaceRecognition_dataset"

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

metadata = load_metadata(path)

classifier = Classifier(method='svc')

'''This is for training'''
#classifier.trainClassifier(metadata)

from glob import glob
#img_mask = path + '/**/' + '*.jpg'
user_names = glob(path+"/*")

#cam = cv2.VideoCapture('/Users/siddhartham/Sid/PycharmProjects/CNN-face-recognition/id-face_turn.MOV')
#cam = cv2.VideoCapture(0)
imagePath = '/Users/siddhartham/Sid/PycharmProjects/Deep-Face-Recognition-Using-Inception-Model-Keras-OpenCV-Dlib/Edmund_photos/Edmund_id_2.png'
# Loop
while True:
    # Read the video frame
    #ret, img =cam.read()

    img = cv2.imread(imagePath)

    #img = transformations(img,0)


    # Convert the captured frame into grayscale
    gray = convertToGrayscale(img)
    #print("processing")

    # Get all face from the video frame
    #faces = faceCascade.detectMultiScale(gray, 1.2,5)
    faces = cascadePath(gray, 0)

    print("faces:", faces.__dict__)

    # For each face in faces
    for face in faces:
        print("I am here")
        x = face.left()
        y = face.top()  # could be face.bottom() - not sure
        w = face.right() - face.left()
        h = face.bottom() - face.top()

        #x,y,w,h = face

        x2 = x + w
        y2 = y + h

        face_image = gray[y:y2, x:x2]

        aligned_image = align_image(face_image)

        print(type(aligned_image))
        if type(aligned_image) is not np.ndarray:
            continue
        else:
            #aligned_image = aligned_image[..., ::-1]

            aligned_image = convertToColor(aligned_image)

            #aligned_image = np.rollaxis(aligned_image, axis=0)

            print(aligned_image.shape)

            #cv2.imwrite("/Users/siddhartham/Sid/PycharmProjects/Face-Recognition/Edmund_photos/Hafidz_model_test.jpg", aligned_image)
            #loaded_img = load_image("/Users/siddhartham/Sid/foo/x.jpg")
            #print(loaded_img.shape)



            image_class = classifier.classify(aligned_image)
            print(image_class)


            #image_class, confidence = classifier.classify(aligned_image)
            #accuracy = round(confidence * 100, 2)
            #print(image_class, accuracy)


            # Create rectangle around the face
            cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

            cv2.rectangle(img, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
            #cv2.putText(img, (image_class + " - " + str(accuracy)), (x, y - 40), font, 1, (255, 255, 255), 3)
            cv2.putText(img, (image_class), (x, y - 40), font, 1, (255, 255, 255), 3)

    #cv2.imwrite("/Users/siddhartham/Sid/PycharmProjects/Face-Recognition/Edmund_photos/after_testing_without_accu/Edmund_id_2.png",img)
    cv2.imshow('Image', img)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#2*((precision*recall)/(precision+recall)