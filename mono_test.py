import cv2
from numpy import argmax
import torch
import time
import numpy as np
import serial #pip3 install pyserial
import os, sys
import socket
class turtlebot3():
    def __init__(self):
        self.cap = cv2.VideoCapture(0) 
        model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        # Move model to GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        # Load transforms to resize and normalize the image
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform3 
        else:
            self.transform = midas_transforms.small_transform

    def check_image(self):
        success, img = self.cap.read()
        start = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Apply input transforms
        input_batch = self.transform(img).to(self.device)
        # Prediction and resize to original resolution
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()

        return(img, depth_map)
        
def monodepth():

    turtle = turtlebot3()

    while True:
        img, depth_map = turtle.check_image()

        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        outimg = cv2.hconcat([img, depth_map])
        cv2.imshow('Image', outimg)

        if cv2.waitKey(1) == 27:
            break
        

if __name__ == '__main__':
    monodepth()
 