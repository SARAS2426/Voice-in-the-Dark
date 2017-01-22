# -*- coding: utf-8 -*-

import cv2
import numpy as np


def processImage(img):
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
   
    edgy=cv2.Canny(gray,100,200)        
    
   
    return [gray ,edgy]
