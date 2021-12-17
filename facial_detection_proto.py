import sys
import numpy
import cv2 as cv
face_cascade = cv.CascadeClassifier(#location for cascade for face)
eye_cascade = cv.CascadeClassifier(#fill with eye cascade file location)
cap = cv.VideoCapture(0)
while cap.isOpened():
    falg, img = cap.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("Original:",img)
    cv.imshow("Greyscale:", grey)
    faces = face_cascade.detectMultiScale(img, 1.1, 7)
    eyes = eye_cascade.detectMultiScale(img, 1.1, 7)
    for x,y,w,h in faces:
        cv.rectangle(img, (x,y), (x+ w, y + h), (0,255,0), 1)
    for a,b,c,d in eyes:
        cv.rectangle(img, (a, b), (a+c, b+d), (255,0,0), 1)
    cv.imshow("img", img)
    c = cv.waitKey(1)
    if c == ord("q"):
        break;
cv.release()
cv.destroyAllWindows()