# -*- coding: utf-8 -*-
import p1 as p1
import pickle

def speech_recog():
    
    text=p1.listen()
   # text='watch'
    print("Object Needed="+str(text))
    
    tg_file = open('tg_names.pkl', 'rb')
    tg_names = pickle.load(tg_file)
    tg_file.close()
    
    object_class=tg_names[text]  
    
    return  object_class-1