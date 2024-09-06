import cv2
import math 
from ultralytics import YOLO
from playsound import playsound
import pyttsx3
#ip_camera_address = "http://192.168.0.100:8080"
url=0#"D:\Sudhan\Others\Video of object detection test.mp4"
cap = cv2.VideoCapture(url)
model = YOLO("yolo-Weights/yolov8n.pt")
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
def speech(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
unique_labels = set() 
while True:
    success, img = cap.read()
    results = model(img, stream=True)

  
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])            
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
        
            unique_labels.add(classNames[cls])

    cv2.imshow('IP Camera', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
print(unique_labels)
    
i=0
new_sentence = []
for label in unique_labels:
    if i==0:
        new_sentence.append(f"I found a {label}, and , ")
    else:
        new_sentence.append(f"a {label}")
        
    i+=1 
speech(" ".join(new_sentence))

cv2.destroyAllWindows()
