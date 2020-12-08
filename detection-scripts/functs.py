import cv2
import numpy as np
import functs as ft
from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import PIL

confThreshold = 0.5
nmsThreshold = 0.3
whT = 416
classesFile = 'coco_names.txt'
classNames = []


def findObjects(outputs, img, classNames):
    hT, wT, cT = img.shape
    bbox=[]
    classIds = []
    confs = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+w), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
class objWin():
    def __init__(self):
        self.root = Tk()
        self.root.title("Object Detection - Camera")
        self.root.bind('<Escape>', lambda e: root.quit())
        self.lmain = Label(self.root)
        self.lmain.pack()
        self.btnText = tk.StringVar()
        self.btnText.set("Click to start Object Detection")        
        self.btn = Button(self.root, textvariable=self.btnText, command=self.changeText)
        self.btn.pack(side="bottom", fill="both", expand="no", padx="10", pady="10")
        
    def changeText(self):
        if self.btnText == "Click to start Object Detection":
            self.btnText.set("Click to stop Object Detection")
        else:
            self.btnText.set("Click to start Object Detection")
    
    