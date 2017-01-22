# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import scipy.optimize as sp_opt
import cv2
from sklearn.cross_validation import train_test_split 
from sklearn.preprocessing import normalize
import time
from sys import stdout
import ProcessImage as PI 

completion=2          

def detectObject(frame):
    
     import numpy as np
     
     print("DETECTING OBJECT")  
    
     img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     #th3=cv2.Canny(img,100,200)
     #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
     res=cv2.resize(img,(70,70),interpolation=cv2.INTER_AREA)
     tg_file = open('tg_names.pkl', 'rb')
     tg_names = pickle.load(tg_file)
     tg_file.close()
    
     wt1=np.loadtxt('wt1')
     wt2=np.loadtxt('wt2')
     wt3=np.loadtxt('wt3')
       
     X=np.array(np.ravel(res))
     X=X.reshape((1,len(X)))
     X=np.array(normalize(X,norm='l2',axis=1))
     X=np.ravel(X)
     
     X_test=np.ones((1,X.shape[0]+1))
     X_test[0,1:]=X
      
       
     z2=X_test.dot(wt1)
     a2=np.ones((z2.shape[0],z2.shape[1]+1))    
     a2[:,1:]=sigmoid(z2)
             
     z3=a2.dot(wt2)
     a3=np.ones((z3.shape[0],z3.shape[1]+1))    
     a3[:,1:]=sigmoid(z3)
     
     z4=a3.dot(wt3)
     a4=sigmoid(z4)
     a4=np.round(a4,3)
    
     
     print(tg_names) 
    
     print(str(a4*100))
     perf=a4*100
     p=np.array(np.where(perf==np.max(perf))) 
       
     print(p)   
     
     if np.max(a4*100)>50 :       
        return [True,perf]
     else :
        return [False,-1]
     
     
    

class NN:
    def __init__(self,l1_size,l4_size):
        self.l1_size=l1_size
        self.l2_size=200
        self.l3_size=50
        self.l4_size=l4_size
        
        self.wt1=np.random.normal(0,0.5,(self.l1_size+1,self.l2_size))      
        self.wt2=np.random.normal(0,0.5,(self.l2_size+1,self.l3_size))
        self.wt3=np.random.normal(0,0.5,(self.l3_size+1,self.l4_size))
        
        
        
        self.cost=10000000
        self.learning_algorithm="BATCH GRADIENT DESCENT"
        self.alpha=1
        self.dropout_percent=0.05
        self.do_dropout=False
        
    def fit(self,X,Y,epochs):
         print("TRAINING NETWORK...")
         X_train=np.ones((X.shape[0],X.shape[1]+1))
         X_train[:,1:]=X
         Y_train=Y
         
         self.learn(X_train,Y_train,epochs)
         
       
    def learn(self,X_train,Y_train,epochs):
        print("wts are converging")
        
        init_wt=np.append(self.wt1,self.wt2)
        init_wt=np.append(init_wt,self.wt3)
        _lambda=100
       
        self.alpha=0.05
        
          
        opts = {'maxiter' :epochs, 'disp':True}
        T=sp_opt.minimize(self.backprop, x0=init_wt, jac=True, args=(X_train,Y_train,_lambda), method='CG',options=opts)
        wt=T.x
        self.wt1=np.reshape(wt[0:self.l2_size*(self.l1_size+1)],(self.l1_size+1,self.l2_size))
        self.wt2=np.reshape(wt[self.l2_size*(self.l1_size+1):self.l2_size*(self.l1_size+1)+self.l3_size*(self.l2_size+1)],(self.l2_size+1,self.l3_size))
        self.wt3=np.reshape(wt[self.l2_size*(self.l1_size+1)+self.l3_size*(self.l2_size+1):],(self.l3_size+1,self.l4_size))
        
        
 
        
   
         
         
         
    def backprop(self,init_w,X,Y,_lambda):
        
        global completion
        
        
        m=len(X)
       
        Y=Y.reshape((len(Y),1)) 
        w1=np.reshape(init_w[0:(self.l1_size+1)*self.l2_size],(self.l1_size+1,self.l2_size))
        w2=np.reshape(init_w[(self.l1_size+1)*self.l2_size:(self.l1_size+1)*self.l2_size+(self.l2_size+1)*self.l3_size],(self.l2_size+1,self.l3_size))
        w3=np.reshape(init_w[(self.l1_size+1)*self.l2_size+(self.l2_size+1)*self.l3_size:],(self.l3_size+1,self.l4_size))
      
          
        
        z2=X.dot(w1)
            
        a2=np.ones((z2.shape[0],z2.shape[1]+1))    
        a2[:,1:]=sigmoid(z2)
        
        
#        dropout_matrix1=np.ones(a2.shape[1]-1)
#        if self.do_dropout==True:
#            dropout_matrix1=np.random.binomial(1, 0.5, size=(a2.shape[1]-1))
#            
#           # dropout_matrix1=np.reshape(dropout_matrix1,(1,len(dropout_matrix1)))   
#            a2[:,1:]*=dropout_matrix1*(1/0.5)
#       
        

        z3=a2.dot(w2)
      
        a3=np.ones((z3.shape[0],z3.shape[1]+1))
        a3[:,1:]=sigmoid(z3)
        
#        dropout_matrix2=np.ones(a3.shape[1]-1)
#        if self.do_dropout==True:
#           dropout_matrix2=np.random.binomial(1, 0.8, size=(a3.shape[1]-1))                     
#           a3[:,1:]*=dropout_matrix2*(1/0.8)
#        
        
        z4= a3.dot(w3)
       
        a4=sigmoid(z4)        
        

         
         
        lg1=np.log(a4)
        lg2=np.log(1-a4)
        
        
        y=np.zeros(lg1.shape)
       
        for i in range(lg1.shape[0]):
            y[i,Y[i]-1]=1
            
  
            
        val1=y*lg1
        val2=(1-y)*lg2
        val=-val1-val2
        
        J=np.sum(val)/(m)
        correction=(np.sum(w1[:,1:]**2)+np.sum(w2[:,1:]**2)+np.sum(w3[:,1:]**2))*(_lambda/(2*m))
        J=J+correction    
        
        s4=a4-y
        s3=s4.dot(w3.T)
        s3=s3[:,1:] 
        s3=s3*sigmoid_gradient(a3[:,1:])
       
        
        s2=s3.dot(w2.T)
        s2=s2[:,1:]    
        s2=s2*sigmoid_gradient(a2[:,1:])
      
    
         
        
        del1=np.zeros(w1.shape)
        del2=np.zeros(w2.shape)
        del3=np.zeros(w3.shape)
        
        del1=del1+(X.T).dot(s2)
        del2=del2+(a2.T).dot(s3) 
        del3=del3+(a3.T).dot(s4)              
        
        Theta1N=np.zeros(w1.shape)
        Theta1N=w1
        Theta1N[:,0]=0
        
        Theta2N=np.zeros(w2.shape)
        Theta2N=w2
        Theta2N[:,0]=0
        
        Theta3N=np.zeros(w3.shape)
        Theta3N=w3
        Theta3N[:,0]=0
        
        Theta1_grad=(del1/m)+(Theta1N)*(_lambda/m)
        Theta2_grad=(del2/m)+(Theta2N)*(_lambda/m)
        Theta3_grad=(del3/m)+(Theta3N)*(_lambda/m)
        
        Theta1_grad=Theta1_grad*self.alpha
        
        Theta2_grad=Theta2_grad*self.alpha
        
#        Theta2_grad[:,dropout_matrix2==0]=0
#        Theta2_grad/=0.8    
#        
        Theta3_grad=Theta3_grad*self.alpha
        
        
        
        grad=np.append(Theta1_grad,Theta2_grad)
        grad=np.append(grad,Theta3_grad)
        
        stdout.flush()
        if J<self.cost:
              print('\r cost:'+str(J)+"    dec(-)    "+str(self.alpha)+"     "),
              stdout.flush()
        else:
              print('\r cost:'+str(J)+"    inc(+)    "+str(self.alpha)),
              stdout.flush()
        self.cost=J
      
        completion+=1
        
        return [J,grad]  
        
        
        
        
    def predict(self,X):
            
             m=len(X)
             X_test=np.ones((X.shape[0],X.shape[1]+1))
             X_test[:,1:]=X
               
             z2=X_test.dot(self.wt1)
             a2=np.ones((z2.shape[0],z2.shape[1]+1))    
             a2[:,1:]=sigmoid(z2)
                         
             z3=a2.dot(self.wt2)
             a3=np.ones((z3.shape[0],z3.shape[1]+1))  
             a3[:,1:]=sigmoid(z3)
             
             z4=a3.dot(self.wt3)
             a4=sigmoid(z4)
             
             pred=np.zeros((m,1))
             for i in range(m):
                    temp=a4[i,:]
                    p=np.where(temp==np.max(temp))        
                    pred[i]=int(p[0]+1)
             
             return pred
                         

def sigmoid(z):

    return 1/(1+np.exp(-z))        
    
def sigmoid_gradient(z):
    
    return z*(1-z)       
        

    
def getTrainData():

        path="C:\\Users\\Akshit Verma\\Desktop\\OCR_VISION\\Test4\\"
        cluster=1
        cluster_list={}
        X1=[]
        X2=[]
        training_data_count={}
        count=0
        for folder in os.listdir(path):
           cluster_list[folder]=cluster
           count=0
           for image in os.listdir(path+folder):
               img=cv2.imread(path+folder+"\\"+image)
               #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
               #th3=cv2.Canny(img,100,200)               
               #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
               P_images=PI.processImage(img)      
               res1=cv2.resize(P_images[0],(70,70),interpolation=cv2.INTER_AREA)
               res2=cv2.resize(P_images[1],(70,70),interpolation=cv2.INTER_AREA)
               X1.append((np.ravel(res1),cluster))
               X2.append((np.ravel(res2),cluster))
               cv2.imshow('IMG',res1)
               cv2.waitKey(1)
               
               count+=1
           training_data_count[folder]=count    
           cluster=cluster+1
        
        cv2.destroyAllWindows()
     
        return [X1,X2,cluster_list,training_data_count]    
              
    
if __name__=='__main__':
                
    [train_frame1,train_frame2,target_names,training_data_count]=getTrainData()
    array1=np.array(train_frame1)     
    X1=np.array(list(array1[:,0]))
    Y1=np.array(list(array1[:,1]))
    
    array2=np.array(train_frame2)   
    X2=np.array(list(array2[:,0]))
    Y2=np.array(list(array2[:,1]))
   
    X1=normalize(X1,norm='l2',axis=1)
    X2=normalize(X2,norm='l2',axis=1)
        
#    X=np.array(np.ones((X1.shape[0]+X2.shape[0],X1.shape[1])))
#    X[0:X1.shape[0],:]=X1
#    X[X2.shape[0]:,:]=X2  
#    
#    Y=np.ones((Y1.shape[0]+Y2.shape[0]))
#    print(Y.shape)
#    Y[0:Y1.shape[0]]=Y1
#    Y[Y1.shape[0]:]=Y2  
#    Y=Y.astype('int')
    
    X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.2, random_state=1)
   
    #X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.2, random_state=1)
        

    nn=NN(X_train.shape[1],len(target_names))
   # nn2=NN(X_train2.shape[1],len(target_names))
    
    wt1=np.zeros((nn.wt1.shape))
    wt2=np.zeros((nn.wt2.shape))
    wt3=np.zeros((nn.wt3.shape))
    
    
    performance_per_epoch=[]
    for epochs in range(120,250,150):
        t1=time.time()
        nn.__init__(X_train.shape[1],len(target_names))
        
        nn.fit(X_train,Y_train,epochs)  
        
        cost1=nn.cost
        print("TIME TO LEARN="+str(time.time()-t1)+"sec")
        t1=time.time()   
        pred_train=nn.predict(X_train)
        pred_train=pred_train.astype(int)
        print("Time to predict="+str(time.time()-t1)+"sec")
        accr_train=np.mean(pred_train.T==Y_train)*100
    
        t1=time.time()
        pred_test=nn.predict(X_test)
        pred_test=pred_test.astype(int)
        print("Time to predict="+str(time.time()-t1)+"sec")
        accr_test=np.mean(pred_test.T==Y_test.T)*100
        print(str(accr_train)+" , "+str(accr_test))
        tg=1
        performance_per_target=[]
        for name in target_names.iterkeys():
            tg=target_names.get(name)
            performance_per_target.append([name,np.mean(pred_test[Y_test==tg]==tg)])
        print(performance_per_target)    
        performance_per_epoch.append([accr_train,accr_test,performance_per_target])    
      
        np.savetxt('wt1',nn.wt1)
        np.savetxt('wt2',nn.wt2)
        np.savetxt('wt3',nn.wt3)
        
        tg_names=open('tg_names.pkl','wb')
        pickle.dump(target_names,tg_names)
        tg_names.close()
        
                
            


        
        
        
    
    
    
    
    
    

    
    














































    