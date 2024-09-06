import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cam = cv.VideoCapture(0)

while True :

    succ , frm = cam.read()

    if succ:

        res = model(frm)
        frm = res[0].plot()
        cv.imshow("ouput",frm)

        if cv.waitKey(1) & 0XFF == ord('q'):
            break
    else:
        break

