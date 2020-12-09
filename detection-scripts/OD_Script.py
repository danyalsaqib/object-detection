import cv2
import numpy as np
import functs as ft
from tkinter import *
import tkinter as tk
from PIL import Image
from PIL import ImageTk
import PIL


cap = cv2.VideoCapture(0)
whT = 416

classesFile = 'coco_names.txt'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    
# print(classNames)
# print(len(classNames))

modelConfiguration = 'yolov4.cfg'
modelWeights = 'yolov4.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

wind = ft.objWin(0, 0, '0', '0')

def Lolv1():
    img = cv2.imread('object-detection-load.png')

    x = 0
    y = 0
    a = "NONE"
    b = "0"

    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    im = PIL.Image.fromarray(cv2image)
    imTk = ImageTk.PhotoImage(im)
    wind.lmain.imTk = imTk
    wind.lmain.configure(image = imTk)
    wind.newVar(x, y, a, b)
    if wind.controlVar:
        wind.lmain.after(1, Lol)
    else:
        wind.lmain.after(1, Lolv1)

def Lol():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0,0,0], 1, crop = False)
    net.setInput(blob)
    
    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    
    outputs = net.forward(outputNames)
    
    x = 0
    y = 0
    a = "NONE"
    b = "0"
    x, y, a, b = ft.findObjects(outputs, img, classNames)

    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    im = PIL.Image.fromarray(cv2image)
    imTk = ImageTk.PhotoImage(im)
    wind.lmain.imTk = imTk
    wind.lmain.configure(image = imTk)
    wind.newVar(x, y, a, b)
    if wind.controlVar:
        wind.lmain.after(1, Lol)
    else:
        wind.lmain.after(1, Lolv1)
        

Lolv1()
wind.root.mainloop()