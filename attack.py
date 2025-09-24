import numpy as np
from model.cnn import create_attack_model_mlp
from model.cnn import create_attack_model_xgboost
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import tensorflow as tf 


class Attack:
    def __init__(self,i,X_train,Y_train,lr,num_labels,model,batch_size,param_grid=None):
        self.name=f"attack-{i}"
        if model=="attack_mlp":
            self.tr=tf.data.Dataset.from_tensor_slices((X_train,Y_train))
            self.train=self.tr.shuffle(self.tr.cardinality()).batch(batch_size,drop_remainder=True)
            self.model=create_attack_model_mlp(lr,num_labels)
        elif model=="attack_xgboost":
            self.x_train=X_train
            self.y_train=Y_train
            self.grid_search=GridSearchCV(XGBClassifier(),param_grid,cv=5)
            # self.model=create_attack_model_xgboost(lr,max_depth,num_estimators)

    def load_model(self,i,flag,folder):
        self.model.load_weights(fr'.\{folder}\{flag}-attack-{i}.h5')
    
    def save_model(self,i,flag,folder,attack_model):
        if attack_model=="attack_mlp":
            self.model.save_weights(fr'.\{folder}\{flag}-attack-{i}.h5')
        # elif attack_model=="attack_xgboost":
        #     self.grid_search.save_model(fr'.\{folder}\{flag}-attack-{i}.json')
        
    def local_train(self,attack_model,epochs,verbose):
        if attack_model=="attack_mlp":
            self.model.fit(self.train,epochs=epochs,verbose=verbose)
        elif attack_model=="attack_xgboost":
            self.grid_search.fit(self.x_train,self.y_train,verbose=True)
            print(f"Best Params for {self.name}):",self.grid_search.best_params_)

    def predict(self,data):    
        probabilities=self.model.predict(data)
     
