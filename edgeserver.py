from utils import average_weights
from utils import sum_list
from utils import l2_norm
from utils import multiply
from model.initialize_model import create
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import clone_model


class Edgeserver:          
    def __init__(self,id_name,cnames,dataset,model,loss,metrics,lr,image_shape,num_labels):         
        n='edge'
        self.name=f'{n}_{id_name+1}'
        self.cnames=cnames
        self.buffer={}   #  clients updates (in learning and unlearning steps)
        self.participated_sample={}
        self.model=create(dataset,model,loss,metrics,lr,image_shape,num_labels)   # in learning and unlearning steps
        #self.updates_buffer={}        
        self.calibrating_cnames=None

    def aggregate(self,global_r=None,community_r=None,folder=None,flag=None): 
        if flag=="saving update": 
            temp_model=clone_model(self.model)
            for client_name,client_update in self.buffer.items():       #❌ یا global_r+1?  community_r+1?
                temp_model.set_weights(client_update)
                temp_model.save_weights(fr".\{folder}\itr_{global_r}_agg_{community_r}_{client_name}.h5") 
        sample_number=[]
        weight=[]
        for i in self.participated_sample.values():          
            sample_number.append(i)
        for w in self.buffer.values():
            weight.append(w)
        final_weights=sum_list(self.model.get_weights(),average_weights(w=weight,sample_num=sample_number))
        self.model.set_weights(final_weights) 
        #self.model.save_weights(fr".\{folder_5}\itr_{global_r}\agg_{community_r}_{self.name}.h5")   # یا global_r+1?  community_r+1?

    def calibrate_and_aggregate_weights(self,global_r,community_r,folder):
        temp_model=clone_model(self.model)
        sample_number=[]
        for i in self.participated_sample.values():          
            sample_number.append(i)
        for client_name in self.calibrating_cnames:
            temp_model.load_weights(fr".\{folder}\itr_{global_r}_agg_{community_r}_{client_name}.h5") 
            ratio=l2_norm(temp_model.get_weights())/l2_norm(self.buffer[client_name])  #ratio in update calibrating
            self.buffer[client_name]=multiply(ratio,self.buffer[client_name])   #update calibrating
        weight=[]
        for w in self.buffer.values():
            weight.append(w)
        final_weights=sum_list(self.model.get_weights(),average_weights(w=weight,sample_num=sample_number))
        self.model.set_weights(final_weights)
        
    def send_to_client(self,client): 
        client.model.set_weights(self.model.get_weights())

    def send_to_server(self,server):
        server.buffer[self.name]=self.model.get_weights()   
                                
    def refresh_edgeserver_buffer(self):                                               
        self.buffer.clear()
        
    def refresh_edgeserver_size(self):                                               
        self.participated_sample.clear()
                  
    def client_registering(self,client):    
        self.participated_sample[client.name]=client.train_num
       
    def m_compile(self,loss,optimizer,metrics):    
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
