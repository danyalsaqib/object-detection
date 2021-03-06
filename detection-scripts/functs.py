import cv2
import numpy as np
import functs as ft
from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import PIL

confThreshold = 0.4
nmsThreshold = 0.3
whT = 416
classesFile = 'coco_names.txt'
classNames = []



def findObjects(outputs, img, classNames):
    hT, wT, cT = img.shape
    bbox=[]
    classIds = []
    confs = []
    x = 0
    y = 0
    a = []
    b = []
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
        lrg = np.argmax(confs)
        a = f'{classNames[classIds[lrg]].upper()}'
        b = f'{int(confs[lrg]*100)}%'
    return x, y, a, b
        
class objWin():
    def __init__(self, x, y, a, b):
        self.root = Tk()
        self.root.title("Object Detection - Camera")
        self.root['bg'] = '#565656'
        self.root.bind('<Escape>', lambda e: root.quit())
        self.canvas1 = tk.Canvas(self.root, width = 400, height = 300,  
                                 relief = 'raised')
        self.lmain = Label(self.root, bd=2, relief=FLAT)
        self.lmain.grid(row = 0, column = 0, sticky = W, pady = 2,
                        rowspan = 8, columnspan = 3) 
        self.btnText = tk.StringVar()
        self.xvar = tk.StringVar()
        self.yvar = tk.StringVar()
        self.avar = tk.StringVar()
        self.bvar = tk.StringVar()
        self.xvar.set(str(x))
        self.yvar.set(str(y))
        self.avar.set(a)
        self.bvar.set(b)
        
        #self.btnText.set("Click to start Object Detection")
        self.lol1 = 0        
        self.controlVar = False
        self.btn = Button(self.root, text="Click to start Object Detection", 
                          command=self.changeText, height=3, width=30)
        self.btn.config(bg = '#DCDCDC')
        self.btn.grid(row = 7, column = 5, pady = 2,
                      columnspan = 1)
        
        self.labela1 = tk.Label(self.root, text = "Detected Object: ",
                                fg = 'white')
        self.labela1.config(bg = '#565656')
        self.labela2 = tk.Label(self.root, textvariable=self.avar,
                                 height=2, width=15)
        self.labela2.config(bg = 'white')
        self.labela1.grid(row = 0, column = 4, sticky = W, pady = 10) 
        self.labela2.grid(row = 0, column = 6, sticky = W, pady = 10,
                          columnspan = 2) 
        
        self.labelb1 = tk.Label(self.root, text = "Confidence (%): ",
                                fg = 'white')
        self.labelb1.config(bg = '#565656')
        self.labelb2 = tk.Label(self.root, textvariable=self.bvar,
                                 height=2, width=15)
        self.labelb2.config(bg = 'white')
        self.labelb1.grid(row = 1, column = 4, sticky = W, pady = 10) 
        self.labelb2.grid(row = 1, column = 6, sticky = W, pady = 10,
                          columnspan = 2) 
        
        self.labelx1 = tk.Label(self.root, text = "X Co-ordinate: ",
                                fg = 'white')
        self.labelx1.config(bg = '#565656')
        self.labelx2 = tk.Label(self.root, textvariable=self.xvar,
                                height=2, width=15)
        self.labelx2.config(bg = 'white')
        self.labelx1.grid(row = 2, column = 4, sticky = W, pady = 10) 
        self.labelx2.grid(row = 2, column = 6, sticky = W, pady = 10,
                          columnspan = 2) 
        
        self.labely1 = tk.Label(self.root, text = "Y Co-ordinate: ",
                                fg = 'white')
        self.labely1.config(bg = '#565656')
        self.labely2 = tk.Label(self.root, textvariable=self.yvar,
                                 height=2, width=15)
        self.labely2.config(bg = 'white')
        self.labely1.grid(row = 3, column = 4, sticky = W, pady = 10) 
        self.labely2.grid(row = 3, column = 6, sticky = W, pady = 10,
                          columnspan = 2) 
        
    def changeText(self):
        if self.lol1 == 0:
            self.btn.config(text = "Click to stop Object Detection",
                            bg = '#a9a9a9')
            self.lol1 = 1
            self.controlVar = True
        else:
            self.btn.config(text = "Click to start Object Detection",
                            bg = '#DCDCDC')
            self.lol1 = 0
            self.controlVar = False
            
    def newVar(self, x, y, a, b):
        self.xvar.set(str(x))
        self.yvar.set(str(y))
        self.avar.set(a)
        self.bvar.set(b)
    