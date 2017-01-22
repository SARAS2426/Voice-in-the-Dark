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

completion=2          

def detectObject(frame):
    
     import numpy as np
     
     print("DETECTING OBJECT")  
     img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     th3=cv2.Canny(img,100,200)
     #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
     res=cv2.resize(th3,(50,50),interpolation=cv2.INTER_AREA)
     #plt.imshow(res)
     #plt.show()
     tg_file = open('tg_names.pkl', 'rb')
     tg_names = pickle.load(tg_file)
     tg_file.close()
    
     wt1=np.loadtxt('wt1')
     wt2=np.loadtxt('wt2')
       
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
     a3=sigmoid(z3)
     a3=np.round(a3,3)
     
    # for name in tg_names.iterkeys():
     #    print(name+":"+" "+str(a3[tg_names[name]-1]))
     
     
     print(tg_names) 
    
     print(str(a3*100))
     p=np.array(np.where(a3==np.max(a3))) 
     
     
       
     print(p)   
     
    
  
#
#     if a3[p[0]]>20  :
#         return "True"
#     else:
     return False
         

class NN:
    def __init__(self,l1_size,l3_size):
        self.l1_size=l1_size
        self.l2_size=200
        self.l3_size=l3_size
        
        self.wt1=np.random.normal(0,0.2,(self.l1_size+1,self.l2_size))      
        self.wt2=np.random.normal(0,0.2,(self.l2_size+1,self.l3_size))
        
        self.cost=10000000
        self.learning_algorithm="BATCH GRADIENT DESCENT"
        self.alpha=1
        self.dropout_percent=0.2
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
        _lambda=200
       
        self.alpha=0.05
        
        
        opts = {'maxiter' :epochs, 'disp':True}
        T=sp_opt.minimize(self.backprop, x0=init_wt, jac=True, args=(X_train,Y_train,_lambda), method='CG',options=opts)
        wt=T.x
        self.wt1=np.reshape(wt[0:self.l2_size*(self.l1_size+1)],(self.l1_size+1,self.l2_size))
        self.wt2=np.reshape(wt[self.l2_size*(self.l1_size+1):],(self.l2_size+1,self.l3_size))  
        
    def predict(self,X):
        
         m=len(X)
         X_test=np.ones((X.shape[0],X.shape[1]+1))
         X_test[:,1:]=X
           
         z2=X_test.dot(self.wt1)
         a2=np.ones((z2.shape[0],z2.shape[1]+1))    
         a2[:,1:]=sigmoid(z2)
                     
         z3=a2.dot(self.wt2)
         a3=sigmoid(z3)
         
         pred=np.zeros((m,1))
         for i in range(m):
                temp=a3[i,:]
                p=np.where(temp==np.max(temp))        
                pred[i]=int(p[0]+1)
         
         return pred
                 
         
         
         
    def backprop(self,init_w,X,Y,_lambda):
        
        global completion
        
        
        m=len(X)
        Y=Y.reshape((len(Y),1)) 
        w1=np.reshape(init_w[0:(self.l1_size+1)*self.l2_size],(self.l1_size+1,self.l2_size))
        w2=np.reshape(init_w[(self.l1_size+1)*self.l2_size:],(self.l2_size+1,self.l3_size))
      
        
        z2=X.dot(w1)
        a2=np.ones((z2.shape[0],z2.shape[1]+1))    
        a2[:,1:]=sigmoid(z2)
        
        if self.do_dropout==True:
            dropout_matrix=np.random.binomial([np.ones(a2.shape)],1-self.dropout_percent)[0][0]
            dropout_matrix=np.reshape(dropout_matrix,(len(dropout_matrix),1))            
            a2*=dropout_matrix
        
        z3=a2.dot(w2)
        a3=sigmoid(z3)
         
         
        lg1=np.log(a3)
        lg2=np.log(1-a3)
        
        y=np.zeros(lg1.shape)
        for i in range(lg1.shape[0]):
            y[i,Y[i]-1]=1
            
  
            
        val1=y*lg1
        val2=(1-y)*lg2
        val=-val1-val2
        
        J=np.sum(val)/(m)
        correction=(np.sum(w1[:,1:]**2)+np.sum(w2[:,1:]**2))*(_lambda/(2*m))
        J=J+correction    
        
        
        s3=a3-y
        s2=s3.dot(w2.T)
        s2=s2[:,1:]    
        s2=s2*sigmoid_gradient(a2[:,1:])
    
         
        
        del1=np.zeros(w1.shape)
        del2=np.zeros(w2.shape)
        
        del1=del1+(X.T).dot(s2)
        del2=del2+(a2.T).dot(s3) 
        
              
        
        Theta1N=np.zeros(w1.shape)
        Theta1N=w1
        Theta1N[:,0]=0
        
        Theta2N=np.zeros(w2.shape)
        Theta2N=w2
        Theta2N[:,0]=0
        
        Theta1_grad=(del1/m)+(Theta1N)*(_lambda/m)
        Theta2_grad=(del2/m)+(Theta2N)*(_lambda/m)
        
        Theta1_grad=Theta1_grad*self.alpha
        Theta2_grad=Theta2_grad*self.alpha
        
        grad=np.append(Theta1_grad,Theta2_grad)
        
        stdout.flush()
        if J<self.cost:
              print('\r cost:'+str(J)+"    dec(-)    "+str(self.alpha)+"     "+str(completion)),
              stdout.flush()
        else:
              print('\r cost:'+str(J)+"    inc(+)    "+str(self.alpha)),
              stdout.flush()
        self.cost=J
       
        completion+=1
        
        return [J,grad]    
            

def sigmoid(z):

    return 1/(1+np.exp(-z))        
    
def sigmoid_gradient(z):
    
    return z*(1-z)       
        

    
def getTrainData():

        path="C:\\Users\\Akshit Verma\\Desktop\\OCR_VISION\\Test4\\"
        cluster=1
        cluster_list={}
        X=[]
        training_data_count={}
        count=0
        for folder in os.listdir(path):
           cluster_list[folder]=cluster
           count=0
           for image in os.listdir(path+folder):
               img=cv2.imread(path+folder+"\\"+image)
               img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
               th3=cv2.Canny(img,100,200)               
               #th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
               res=cv2.resize(th3,(100,100),interpolation=cv2.INTER_AREA)
               X.append((np.ravel(res),cluster))
               cv2.imshow('IMG',res)
               cv2.waitKey(1)
               count+=1
           training_data_count[folder]=count    
           cluster=cluster+1
        
        cv2.destroyAllWindows()
     
        return [X,cluster_list,training_data_count]    
              
    
if __name__=='__main__':
                
    [train_frame,target_names,training_data_count]=getTrainData()
    array=np.array(train_frame)     
    X=np.array(list(array[:,0]))
    Y=np.array(list(array[:,1]))
   
    X=normalize(X,norm='l2',axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    
   
        

    nn=NN(X_train.shape[1],len(target_names))
    
    
    performance_per_epoch=[]
    for epochs in range(30,60,50):
        t1=time.time()
        nn.__init__(X_train.shape[1],len(target_names))
        nn.fit(X_train,Y_train,epochs)  
        cost=nn.cost
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
        tg_names=open('tg_names.pkl','wb')
        pickle.dump(target_names,tg_names)
        tg_names.close()
        


            
        
        
    
    
    
    
    
    

    
    














































    