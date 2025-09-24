from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
import tensorflow as tf
from xgboost import XGBClassifier


def Mnist_CNN_1(loss,metrics,lr,image_shape,num_labels):      # book       Total params: 93,322   for mnist
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    #adam_opt=tf.keras.optimizers.Adam(learning_rate=lr)
    pt = tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=pt)
    #print(model.summary())    
    return model

def Mnist_CNN_2(loss,metrics,lr,image_shape,num_labels):           # communication        Total params: 1,663,370  for mnist
    model = Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation='relu',padding='same', input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))  
    adam_opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=adam_opt)
    
    return model

def Mnist_CNN_3(loss,metrics,lr,image_shape,num_labels):   # on silos       Total params: 107,786   for mnist 
    model=Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='relu',padding='same', input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_labels, activation='softmax'))
    adam_opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=adam_opt)
    return model

def Mnist_CNN_federaser(loss,metrics,lr,image_shape,num_labels):  # Total params:431,080     in federaser paper 
    model=Sequential()
    model.add(layers.Conv2D(20,(5, 5),strides=1,activation='relu',input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(50,(5, 5),strides=1,activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dense(num_labels,activation='softmax'))
    adam_opt=tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=adam_opt)
    return model
    
def Cifar10_CNN_federaser(loss,metrics,lr,image_shape,num_labels):    # Total params:62,006     in federaser paper 
    model=Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=image_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(num_labels,activation='softmax'))   
    pt=tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=pt)
    return model

def Cifar10_CNN_2(loss,metrics,lr,image_shape,num_labels):      # Total params:2,609,034
    model=Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(512, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_labels, activation='softmax'))
    pt=tf.keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss=loss,metrics=metrics,optimizer=pt)
    return model

def create_attack_model_mlp(lr,num_labels):
    model=Sequential()
    model.add(Dense(64,activation='relu',input_shape=(num_labels,))) 
    model.add(Dense(2,activation='softmax'))
    pt=tf.keras.optimizers.Adam(learning_rate=lr)     #🟪
    # model.compile(loss='categorical_crossentropy',metrics=['accuracy','val_accuracy'],optimizer=pt)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=pt)
    return model


def create_attack_model_xgboost(param_grid):
    model=XGBClassifier(n_estimators=num_estimators,max_depth=max_depth,learning_rate=lr,objective='binary:logistic'
          ,tree_method="hist",device="cuda")
    return model

# def create_attack_model_xgboost(lr,num_labels):
