import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8x-pose.pt")
results = model("ex1.jpg", save=True,
save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data)
path = "ex1.jpg"
img = cv2.imread(path)

for i in range(5,17):
    center_coordinates = (keypoints[i].x,keypoints[i].y)#5
    radius = 50
    color=(0,255,0)
    thickness = 1
    cv2.circle(img,center_coordinates,radius,color,thickness)

