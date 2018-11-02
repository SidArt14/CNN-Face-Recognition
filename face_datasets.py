
# Import OpenCV2 for image processing
import cv2
import os, numpy as np

from align import AlignDlib


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

alignment = AlignDlib('/Users/siddhartham/Sid/PycharmProjects/Deep-Face-Recognition-Using-Inception-Model-Keras-OpenCV-Dlib/models/landmarks.dat')

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

from glob import glob
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def run(face_id, name):
    # Start capturing video
    vid_cam = cv2.VideoCapture(0)
    #vid_cam = cv2.VideoCapture('/Users/siddhartham/Sid/PycharmProjects/Face-Recognition/FaceRecognition_Videos/Hafidz.mp4')

    # Detect object in video stream using Haarcascade Frontal Face
    face_detector = cv2.CascadeClassifier('/Users/siddhartham/Sid/PycharmProjects/Deep-Face-Recognition-Using-Inception-Model-Keras-OpenCV-Dlib/haarcascades/haarcascade_frontalface_default.xml')

    # For each person, one face id
    #face_id = 1

    # Initializt sample face image
    count = 0

    assure_path_exists("FaceRecognition_dataset/")
    dir_name = 'FaceRecognition_dataset/%s-%s' % (face_id, name)
    os.makedirs(dir_name)

    print("Starting to capture..sit upright.")
    # Start looping
    while (vid_cam.isOpened()):
        # Capture video frame
        _, image_frame = vid_cam.read()
        image_frame = transformations(image_frame,0)

        # Convert frame to grayscale
        if image_frame is None:

            continue
        #print("captured.")
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

        #gray = cv2.resize(gray, (100, 100))

        # Detect frames of different sizes, list of faces rectangles
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if not len(faces):
            continue

        # Loops for each faces
        for (x, y, w, h) in faces:
            print("I am here")
            # Crop the image frame into rectangle
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            #image_frame = cv2.resize(image_frame, (100,100))

            # Increment sample face image

            # Save the captured image into the datasets folder
            img = align_image(gray[y:y + h, x:x + w])
            print(type(img))
            if type(img) is not np.ndarray:
                continue
            else:
                print("I am here too")
                cv2.imwrite(("%s/User."%dir_name) + str(face_id) + '.' + str(count) + ".jpg", img)
                count += 1

            # Display the video frame, with bounded rectangle on the person's face
                cv2.imshow('frame', image_frame)
                print("already written")
            #else:
                #continue
        # To stop taking video, press 'q' for at least 100ms
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

        # If image taken reach 100, stop taking video
        elif count > 100:
            break

    # Stop video
    vid_cam.release()

    # Close all started windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    faces = glob('FaceRecognition_dataset/*')
    faces = [int(face.split('/')[-1].split('-')[0]) for face in faces]
    face_id = 1
    faces.sort()
    print(faces)
    if len(faces) > 0:
        face_id = faces[-1] + 1
        print(face_id)
    name = input("Enter your name: ")
    print(faces)
    run(face_id, name)
