import cv2
import numpy as np


class ObjectDetection:
    def __init__(self, weights_path="models/yolov3.weights", cfg_path="models/yolov3.cfg"):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv3")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        self.model = cv2.dnn_DetectionModel(net)
        self.classes = []
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()
            #self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)