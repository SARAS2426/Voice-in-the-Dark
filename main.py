# -*- coding: utf-8
from threading import Thread
import Queue
import cv2
from ObjectVision import objectVision
from ObjectDetection import objectDetection            
import glob
import os
import time
import Speech_Recog as Sp_Rec

class Resources:
    
    def __init__(self):
        self.read_frame_queue=Queue.Queue()
        self.new_frame_queue=Queue.Queue()
        self.object_class=-1        



class ObjectVision(Thread):
    def __init__(self,new_frame_queue):
        Thread.__init__(self)
        self.new_frame_queue=new_frame_queue
        print('Started Object Vision System')
        
    def run(self):
      
        if self.new_frame_queue.empty()==False:
            new_frame=self.new_frame_queue.get()
            objectVision(new_frame,res.object_class)
            
        
        




class ObjectDetection(Thread):
    def __init__(self,read_frame_queue,new_frame_queue):
        Thread.__init__(self)
        self.read_frame_queue=read_frame_queue
        self.new_frame_queue=new_frame_queue
        print('Started Object Detection System')
        
        
    def run(self):
        if self.read_frame_queue.empty()==False:
            read_frame=self.read_frame_queue.get()
            self.new_frame_queue.put(objectDetection(read_frame))
        
    
if __name__=='__main__':

 #      INITIALIZING RESOURCES    
    print(" INITIALIZING RESOURCES ")
    res=Resources()
    
  #     INITIALISING OBJECTS   
    print("INITIALISING OBJECTS  ")
    objVision=ObjectVision(res.new_frame_queue)
    objVision.setName("Object Vision System")
    
    objDetect=ObjectDetection(res.read_frame_queue,res.new_frame_queue) 
    objDetect.setName("Object Detection and Navigation")
    
  #     STARTING THREADS OF EXCECUTION TO READ FRAMES 
    print("STARTING THREADS OF EXCECUTION TO READ FRAMES ")
    objVision.start()
    objDetect.start()      
        
    
   #STARTING  CAMERA 
    cap_frames=cv2.VideoCapture(2)
    if cap_frames.isOpened()==False:
        cap_frames.open()
        
        
    print("STARTING  CAMERA ")

    file_index=0
    

    while True:
       
        key=raw_input('press k:')
        if key=='k'   :     
            ret,frame=cap_frames.read()
            cv2.imshow('Reading Frame',frame)
        
        res.object_class=Sp_Rec.speech_recog()
        
       # frame=cv2.imread("hp3.jpg")
        res.read_frame_queue.put(frame)
        

        key=cv2.waitKey(1)  
        if key==ord('q'):
            cv2.destroyAllWindows()  
           # break
        
        if objDetect.isAlive()==False and res.read_frame_queue.empty()==False:     
            objDetect.run()        
      
        if objVision.isAlive()==False and res.new_frame_queue.empty()==False:
            objVision.run()
        
        time.sleep(20)
       
          
        objDetect.join()
        objVision.join()    
        cv2.destroyAllWindows()   
        