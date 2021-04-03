# USB camera display using PyQt and OpenCV, from iosoft.blog
# Copyright (c) Jeremy P Bentham 2019
# Please credit iosoft.blog if you use the information or software in it

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

VERSION = "Face Recognition"

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import warnings
warnings.filterwarnings("ignore")

import sys, time, threading, cv2
try:
    from PyQt5.QtCore import Qt
    pyqt5 = True
except:
    pyqt5 = False
if pyqt5:
    from PyQt5.QtCore import QTimer, QPoint, pyqtSignal
    from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel
    from PyQt5.QtWidgets import QWidget, QAction, QVBoxLayout, QHBoxLayout
    from PyQt5.QtGui import QFont, QPainter, QImage, QTextCursor
else:
    from PyQt4.QtCore import Qt, pyqtSignal, QTimer, QPoint
    from PyQt4.QtGui import QApplication, QMainWindow, QTextEdit, QLabel
    from PyQt4.QtGui import QWidget, QAction, QVBoxLayout, QHBoxLayout
    from PyQt4.QtGui import QFont, QPainter, QImage, QTextCursor
try:
    import Queue as Queue
except:
    import queue as Queue

IMG_SIZE    = 640,480          # 640,480 or 1280,720 or 1920,1080
IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 1                # Scaling factor for display image
DISP_MSEC   = 1                # Delay between display cycles
CAP_API     = cv2.CAP_ANY       # API: CAP_ANY or CAP_DSHOW etc...
EXPOSURE    = 0                 # Zero for automatic exposure
TEXT_FONT   = QFont("Courier", 20)
MTCNN_SCALE = 4               # Scaling factor for mtcnn

camera_num  = 1                 # Default camera (first in list)
image_queue = Queue.Queue()     # Queue to hold images
capturing   = True              # Flag to indicate capturing

anti_spoofing = True            # Apply anti-spoofing

import torch
from torchvision import transforms
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((128,128)),
                            transforms.ToTensor()
                           ])

from mtcnn import MTCNN
from facenet_pytorch import InceptionResnetV1
detector = MTCNN(min_face_size=int(IMG_SIZE[1]/2/MTCNN_SCALE))
resnet = InceptionResnetV1(pretrained='vggface2').eval() # Model of facenet for normal face recognition
model = torch.load('arcface1.pt', map_location='cpu') # our model for masked face recognition

from FaceMaskDetection.mask_detection import inference
from recognize import recognize
from util import *

import requests
url = "http://192.53.175.198/api/face-recognition/" # url of server to request

try:
    import pyrealsense2 as rs
    from depth_cam import DC
    depth_cam = True
    DEPTH_CAM = DC(IMG_SIZE)
    DEPTH_CAM.start()
except Exception as e:
    logging.error(e)
    depth_cam = False
    anti_spoofing = False # turn off anti-spoofing if depth camera is unavailable

# Get face images with MTCNN
def get_face(img):
    temp = detector.detect_faces(cv2.resize(img, (int(IMG_SIZE[0]/MTCNN_SCALE), int(IMG_SIZE[1]/MTCNN_SCALE))))
    box = None
    if len(temp) == 1:  # MTCNN face detected
        box = [MTCNN_SCALE * i for i in temp[0]['box']]
    return box

# Grab images from the camera (separate thread)
def grab_images(cam_num, queue):
    if depth_cam == True:
        try:
            depth_output = pd.DataFrame(columns=['depth'])
            while capturing == True:
                images = DEPTH_CAM.get_frame()
                depth_colormap = DEPTH_CAM.get_depth_colormap(images['depth_image'])
                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = images['color_image'].shape
                if depth_colormap_dim != color_colormap_dim:    # if the depth image dim != color image dim
                    resized_color_image = cv2.resize(images['color_image'], dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)

                if images['color_image'] is not None and queue.qsize() < 2:
                    img = cv2.cvtColor(images['color_image'], cv2.COLOR_BGR2RGB)
                    box = get_face(cv2.resize(img, (int(IMG_SIZE[0]/MTCNN_SCALE), int(IMG_SIZE[1]/MTCNN_SCALE))))
                    v_depth = None
                    if box is not None:  # MTCNN face detected
                        mid_col = round(box[0] + box[2]/2)
                        v_depth = images['depth_image'][box[1]:box[1]+box[3], mid_col:mid_col+1]
                    queue.put((img, box, v_depth))
                else:
                    time.sleep(DISP_MSEC / 1000.0)
        finally:
            # Stop streaming
            DEPTH_CAM.pipeline.stop()
    else:
        cap = cv2.VideoCapture(cam_num-1 + CAP_API)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE[1])
        if EXPOSURE:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        while capturing:
            if cap.grab():
                retval, img = cap.retrieve(0)
                if img is not None and queue.qsize() < 2:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    box = get_face(cv2.resize(img, (int(IMG_SIZE[0]/MTCNN_SCALE), int(IMG_SIZE[1]/MTCNN_SCALE))))
                    queue.put((img, box, None))
                else:
                    time.sleep(DISP_MSEC / 1000.0)
            else:
                print("Error: can't grab camera image")
                break
        cap.release()

# Image widget
class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()

# Main window
class MyWindow(QMainWindow):
    text_update = pyqtSignal(str)

    # Create main window
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.central = QWidget(self)
        self.textbox = QTextEdit(self.central)
        self.textbox.setReadOnly(True)
        self.textbox.setFont(TEXT_FONT)
        self.textbox.setMinimumSize(300, 50)
        self.text_update.connect(self.update_text)
        sys.stdout = self
        if depth_cam == True:
            cam_info = "Using 3D camera"
        else:
            cam_info = "Using normal camera (anti-spoofing unavailable)"

        self.vlayout = QVBoxLayout()        # Window layout
        self.displays = QHBoxLayout()
        self.disp = ImageWidget(self)
        self.displays.addWidget(self.disp)
        self.vlayout.addLayout(self.displays)
        self.label = QLabel(cam_info, self)
        self.label.setFont(TEXT_FONT)
        self.vlayout.addWidget(self.label)
        self.vlayout.addWidget(self.textbox)
        self.central.setLayout(self.vlayout)
        self.setCentralWidget(self.central)

        self.mainMenu = self.menuBar()      # Menu bar
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.mainMenu.addAction(exitAction)

        self.mask = 2
        self.detect_mask_count = 0
        self.recognize_count = 0
        self.recognize_result = None

    # Start image capture & display
    def start(self):
        self.timer = QTimer(self)           # Timer to trigger display
        self.timer.timeout.connect(lambda:
                    self.show_image(image_queue, self.disp, DISP_SCALE))
        self.timer.start(DISP_MSEC)
        self.capture_thread = threading.Thread(target=grab_images,
                    args=(camera_num, image_queue))
        self.capture_thread.start()         # Thread to grab images

    # Fetch camera image from queue, and display it
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            img, box, depth = imageq.get()
            if img is not None and len(img) > 0:

                if box is not None and (anti_spoofing == False or np.squeeze(depth).size > 5):

                    if anti_spoofing:
                        MSE = check_depth(depth)

                    if anti_spoofing and MSE < 3: # fake, rendering red bounding box
                        img = cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255,0,0), 5)
                        print("Warning")
                        self.mask = 2
                        self.detect_mask_count = 0
                        self.recognize_count = 0
                        self.recognize_result = None
                    elif anti_spoofing and MSE > 15: # distance too closed to camera, anti-spoofing fails, rendering orange box
                        img = cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255,165,0), 5)
                        print("Please keep around 0.6m away from the camera")
                        self.mask = 2
                        self.detect_mask_count = 0
                        self.recognize_count = 0
                        self.recognize_result = None
                    else:
                        if self.detect_mask_count % 10 == 0:
                            temp_mask = inference(img, target_shape=(360, 360))
                            if temp_mask is not None:
                                self.mask = temp_mask[1]
                        self.detect_mask_count += 1

                        if self.recognize_count % 10 == 0:
                            if box[0] < 0:
                                box[2] += box[0]
                                box[0] = 0
                            if box[1] < 0:
                                box[3] += box[1]
                                box[1] = 0
                            tensor = norm(trans(img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]).float()).unsqueeze(0)

                            is_masked = False
                            if self.mask == 0:
                                is_masked = True
                                with torch.no_grad():
                                    embed = model(tensor)['embeddings']
                            else:
                                with torch.no_grad():
                                    embed = resnet(tensor)

                            encoded_embedding = embedding_to_base64_string(embed)

                            result = requests.post(url, json={"is_masked": is_masked, "embedding": encoded_embedding})
                            if result is not None and result.json()['is_user'] == True:
                                self.recognize_result = result.json()['user']['first_name'] + " " + result.json()['user']['last_name']
                            else:
                                self.recognize_result = None

                        add_bounding_box(img, box, self.mask, self.recognize_result)
                        self.recognize_count += 1

                        if self.mask == 1:
                            print("Please put on a mask")
                        elif self.mask == 0:
                            if self.recognize_result is not None:
                                print("Welcome " + self.recognize_result)

                elif self.mask != 2:
                    self.mask = 2
                    self.detect_mask_count = 0
                    self.recognize_count = 0
                    self.recognize_result = None
                else:
                    print(" ")

                self.display_image(img, display, scale)

    # Display an image, reduce size if required
    def display_image(self, img, display, scale=1):
        disp_size = img.shape[1]//scale, img.shape[0]//scale
        disp_bpl = disp_size[0] * 3
        if scale > 1:
            img = cv2.resize(img, disp_size,
                             interpolation=cv2.INTER_CUBIC)
        qimg = QImage(img.data, disp_size[0], disp_size[1],
                      disp_bpl, IMG_FORMAT)
        display.setImage(qimg)

    # Handle sys.stdout.write: update text display
    def write(self, text):
        self.text_update.emit(str(text))
    def flush(self):
        pass

    # Append to text display
    def update_text(self, text):
        if str(text) != "\n":
            self.textbox.setText(str(text))

    # Window is closing: stop video capture
    def closeEvent(self, event):
        global capturing
        capturing = False
        self.capture_thread.join()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        try:
            camera_num = int(sys.argv[1])
        except:
            camera_num = 0
    if camera_num < 1:
        print("Invalid camera number '%s'" % sys.argv[1])
    else:
        app = None
        app = QApplication(sys.argv)
        win = MyWindow()
        win.show()
        win.setWindowTitle(VERSION)
        win.start()
        sys.exit(app.exec_())
