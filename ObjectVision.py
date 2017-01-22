import numpy as np
import cv2
import matplotlib.pyplot as plt
from threading import Thread
import Queue
import NeuralNetwork2 as NN
import threading
import calc_dist as c_d


class FrameQueue:
    def __init__(self):
        self.frame_queue=Queue.Queue()
        
class NeuralNetwork(Thread):
    def __init__(self,f_q,object_class):
        self.f_q=f_q
        self.object_class=object_class
        self.lock=threading.Lock()
        Thread.__init__(self)
    
    def run(self):
        
        width=[6,6,6,6]
        cnt=0
        if self.f_q.empty()==False: 
            print((self.f_q.qsize()))  
            self.lock.acquire()
           
               
            img=self.f_q.get()
            #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            img=cv2.resize(img,(800,800),interpolation=cv2.INTER_AREA)
            row_max=img.shape[0]
            col_max=img.shape[1]
            
            
            
            row=0
            col=0
            col_stride=col_max/8
            row_stride=row_max/8
            size_of_segment=600
            
            while row<row_max-size_of_segment:
                col=0
                while col<col_max-size_of_segment:
                    
                    frame=img[row:row+size_of_segment,col:col+size_of_segment,:]
                    
                    col=col+col_stride
                    (msg,obj)=NN.detectObject(frame)
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    cv2.imshow('img',frame)
                    cv2.waitKey(1)
                    if msg==True:   
                        if obj[0,self.object_class]>50:
                            if np.max(obj)==obj[0,self.object_class]:
                                plt.imshow(frame)
                                plt.show()
                            cnt+=1
                    
                        
                row=row+row_stride
            c_d.calc_dist(img,frame,cnt,width[self.object_class],col_stride,obj)    
            
        self.lock.release()    




def objectVision(frame,object_class):

     print(object_class) 
      
     f_q=FrameQueue()
     f_q.frame_queue.put(frame)    
     nn=NeuralNetwork(f_q.frame_queue,object_class)
     nn.start()
     #cv2.imshow('Read Frame',frame)
     #cv2.waitKey(1)
    
     nn.join()
    
   
