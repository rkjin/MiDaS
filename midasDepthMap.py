import cv2
from numpy import argmax
import torch
import time
import numpy as np
import serial #pip3 install pyserial
import os, sys
import socket
class turtlebot3():
    def __init__(self,free):
        self.free = free
        os.system("sudo chmod 777 /dev/ttyACM0")
        self.ser = serial.Serial("/dev/ttyACM0", 115200, timeout = 1)  
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

    def move(self, op):
        self.ser.write(op.encode('utf-8'))
        self.ser.flush() # wait until com port finish job
        if op == 'i':  
            imu = []
            for i in range(9):
                raw  = self.ser.readline()
                raw_data = raw.decode()[:len(raw)-2]
                imu.append(float(raw_data))
            print('AccX :', imu[0])
            print('AccY :', imu[1])
            print('AccZ :', imu[2])
            print('GyrX :', imu[3])                                              
            print('GyrY :', imu[4])
            print('GyrZ :', imu[5])
            print('Roll :', imu[6])
            print('Pitch :', imu[7])
            print('Yaw :', imu[8])
        return (imu)

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
        d_inv = 30000 / ( depth_map + 1e-9)
        d_inv = np.clip(d_inv, 0, 255).astype(np.uint8)
        d_inv = cv2.blur(d_inv, (5,5))
        print('length of 30',len(d_inv[d_inv < 30]))
        if len(d_inv[d_inv < 30])  > 300 :
            is_Clear = 0
        else:
            is_Clear = d_inv.mean()
        return(img, depth_map,d_inv, is_Clear)
        
    def direction(self):
        min_distance = []
        _, _, d_inv, _ = self.check_image()
        if d_inv[:,:320].mean() < d_inv[:,320:].mean():
            dir = True
        else:
            dir = False    
        for i in range(8):
            if dir:                
                os.system('echo l > /dev/ttyACM0')
            else:    
                os.system('echo r > /dev/ttyACM0')
            time.sleep(1)    
            _, _, _, is_Clear = self.check_image()
            min_distance.append(is_Clear)
            direction = argmax(min_distance)
        print('min_distance', min_distance)    
        if dir:
            os.system('echo '+'r'*(8-direction)+' > /dev/ttyACM0')
            time.sleep((8-direction))
        else:
            os.system('echo '+'l'*(8-direction)+' > /dev/ttyACM0')
            time.sleep((8-direction))
        return


def monodepth():
    sender = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    free = cv2.imread('/home/bj/data/dnn/MiDaS/free.jpg',cv2.IMREAD_GRAYSCALE)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('/home/bj/data/dnn/MiDaS/output.avi', fourcc, 15, (1280,480))
    turtle = turtlebot3(free)
    mode =""
    s = 0
    while True:
        img, depth_map, d_inv, is_Clear = turtle.check_image()
        print('is_Clear',is_Clear)
        if is_Clear:
            s = 0 
            if mode != 'f':
                mode = 'f'
                os.system('echo f > /dev/ttyACM0')
        else:
            s += 1
            if s > 3:
                mode = 's'
                os.system('echo s > /dev/ttyACM0')
                turtle.direction() 
                s = 0

        messageToSend = 'min :'+str(d_inv.min()) + ' <35 : '+str(len(d_inv[d_inv < 35])) +' <40:'+str(len(d_inv[d_inv < 40]))
        sender.sendto(str.encode(messageToSend), ('192.168.0.21', 7778))

        depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        outimg = cv2.hconcat([img, depth_map])
        print(outimg.shape)
        cv2.imshow('Image', outimg)
        out.write(outimg)
        if cv2.waitKey(1) == 27:
            break
    cap.release()


if __name__ == '__main__':
    monodepth()
   
