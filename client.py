import numpy as np
from model.initialize_model import create
import tensorflow as tf 
from tensorflow.keras.models import clone_model


class Client:    
    def __init__(self,id_client,train_partition,test_partition,dataset,model,loss,metrics,lr,
                                                       batch_size,image_shape,num_labels):
        n='client'
        self.name=f'{n}_{id_client+1}'
        self.x=train_partition
        self.y=test_partition
        self.train_num=train_partition.cardinality().numpy()
        self.test_num=test_partition.cardinality().numpy()
        train_partition=train_partition.shuffle(train_partition.cardinality())
        self.train=train_partition.batch(batch_size,drop_remainder=True)
        self.test=test_partition.batch(32)
        self.model=create(dataset,model,loss,metrics,lr,image_shape,num_labels)     # # in learning and unlearning steps
        self.model_update=clone_model(self.model)    # in learning and unlearning steps 
        self.train_acc=[]
        self.train_label_frequency=[]
        #self.acc=[]
        
    def local_model_train_and_update(self,epochs,verbose):    #,comm_r,community_r,folder)
        current_weights=self.model.get_weights()
        self.model.fit(self.train,epochs=epochs,verbose=verbose)     
        weight_updates=[new_w-current_w for new_w,current_w in zip(self.model.get_weights(),current_weights)]
        self.model_update.set_weights(weight_updates)
        #self.model.save_weights(fr'.\results\clients\{folder}\itr_{comm_r}\agg_{num_agg}_{self.name}.h5')
        
    def send_to_edgeserver (self,edgeserver): 
        edgeserver.buffer[self.name]=self.model_update.get_weights()
        
    def test_c(self):       
        _,acc=self.model.evaluate(self.test,verbose=0)  
        acc=np.round(acc,2)
        #self.acc.append(acc)
    
    def m_compile(self,loss,optimizer,metrics):
        self.model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
    
