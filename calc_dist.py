# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt


def calc_dist(img,frame,cnt,width,col_stride,obj):
    
    
    cv2.imshow('object',frame)
    cv2.waitKey(1)
    plt.imshow(frame)
    plt.show()    
    
    f=443
    
    distance=f*width/(cnt/2+1)*col_stride/2000
    print(distance)

   