import cv2
import numpy as np
from object_detection import ObjectDetection

#net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
#classes = []
#with open('coco.names', 'r') as f:
#    classes = f.read().splitlines()
#print(classes)

# Initialize Object Detection
od = ObjectDetection()  # Default: yolov3
#od = ObjectDetection("models/yolov2-tiny.weights", "models/yolov2-tiny.cfg")

#cap = cv2.VideoCapture("videos/los_angeles.mp4")
#cap = cv2.VideoCapture("videos/softball.mp4")
cap = cv2.VideoCapture("videos/softball2.mp4")
#cap = cv2.VideoCapture("videos/dog.mp4")

while True:
    val, frame = cap.read()
    # if no more frames left in video
    if not val:
        break 
    
    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    idx=0
    font = cv2.FONT_HERSHEY_PLAIN  # Font of the text put above rectangle
    for box in boxes:
        detected_class_label = od.classes[class_ids[idx]]
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, detected_class_label, (x, y+20), font, 2, (255,255,255), 2) 
        idx+=1

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if (key & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()