import os
import cv2 as cv
import numpy as np

def detect_face(in_memory_photo):
    face_cascade = cv.CascadeClassifier(cv.__path__[0] + "C:\Scan_face.png")
    in_memory_photo = in_memory_photo.read()
    nparr = np.fromstring(in_memory_photo, np.uint8)
    img = cv.imdecode(nparr, 4)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(Gray, 1, 5)
    if len(faces) > 0:
        return True
    else:
        return False
with open(os.path.dirname(os.path.abspath(__file__)) + '/Scan_face.png', 'rb') as in_memory_photo:
    is_it_face = detect_face(in_memory_photo)
    print(is_it_face)