from utils import average_weights
from model.initialize_model import create
import tensorflow as tf
from tensorflow.keras.models import clone_model
import numpy as np 

class Server:
    def __init__(self,dataset,model,loss,metrics,lr,image_shape,num_labels,X_test=None,Y_test=None):     
        self.buffer={}
        self.participated_sample={}                      
        self.model=create(dataset,model,loss,metrics,lr,image_shape,num_labels)  
        self.y=tf.data.Dataset.from_tensor_slices((X_test,Y_test))
        self.test_data=self.y.batch(32)
        self.test_acc=[]

    def aggregate(self,global_r=None,community_r=None,folder=None,flag=None):  
        sample_number=[]
        weights=[]
        for i in self.participated_sample.values():
            sample_number.append(i)
        for w in self.buffer.values():
            weights.append(w)
        if flag=="saving update":
            temp_model=clone_model(self.model)
            for edge_name,edge_model in self.buffer.items():                
                temp_model.set_weights(edge_model)
                temp_model.save_weights(fr".\{folder}\itr_{global_r}_agg_{community_r}_{edge_name}.h5")
        self.model.set_weights(average_weights(w=weights,sample_num=sample_number))

    def save_model(self,global_r,folder,flag):
        self.model.save_weights(fr".\{folder}\itr_{global_r}_{flag}.h5")   
        
    def load_model(self,global_r,folder,flag):
        self.model.load_weights(fr".\{folder}\itr_{global_r}_{flag}.h5") 

    def load_edge_model(self,edge_name,global_r,community_r,folder):
        temp_model=clone_model(self.model)
        temp_model.load_weights(fr".\{folder}\itr_{global_r}_agg_{community_r}_{edge_name}.h5")
        self.buffer[edge_name]=temp_model.get_weights()
    
    def send_to_edgeserver(self,edgeserver): 
        edgeserver.model.set_weights(self.model.get_weights())

    def refresh_server_buffer(self):          
        self.buffer.clear() 
        
    def refresh_server_size(self):                 
        self.participated_sample.clear() 
    
    def edgeserver_registering(self,edgeserver):          
        sample_num=[]
        for i in edgeserver.participated_sample.values():
            sample_num.append(i)
        all_sample_num=sum(sample_num)
        self.participated_sample[edgeserver.name]=all_sample_num
        
    def m_compile(self,loss,optimizer,metrics):
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
        
    def test(self,client=None,flag=None):
        if client==None:
            _,acc=self.model.evaluate(self.test_data,verbose=0)
            self.test_acc.append(np.round(acc,2))
            return np.round(acc,2)        
        else: 
            _,acc=self.model.evaluate(client.train,verbose=0)
            if flag==1:   # print
                return np.round(acc,2)
            else:
                client.train_acc.append(np.round(acc,2))

    def predict(self,X_data):
        return self.model.predict(X_data)
                
