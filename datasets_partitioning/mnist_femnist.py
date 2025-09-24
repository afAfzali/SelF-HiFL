# Download the required dataset,split into data , labels

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import sys
import gc
import json


def get_dataset(dataset,model):
        
    if dataset=='mnist':
        if model=='mlp':
            X_train,Y_train,X_test,Y_test=get_mnist_mlp()
            
        elif model in ('cnn1','cnn2','cnn3'):
            X_train ,Y_train,X_test,Y_test=get_mnist_cnn()
            
    elif dataset=='cifar10':
        X_train,Y_train,X_test,Y_test=get_cifar10()
                
    return X_train,Y_train,X_test,Y_test

def get_mnist_mlp():            
    (X_train,Y_train),(X_test,Y_test)=mnist.load_data()
    X_train=X_train.reshape(X_train.shape[0],28*28).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0],28*28).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test
    
def get_mnist_cnn():
    (X_train,Y_train),(X_test,Y_test)=mnist.load_data()
    X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test
    
def plot_mnist(idx):
    X,Y,_,_=get_mnist()                  # X is list,Y is array
    X=np.array(X).reshape(60000,28,28)
    image=X_train[idx]
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(X[idx],cmap=plt.cm.binary)        #interpolation='bicubic'
    ax.set_title('label ={}'.format(Y[idx]),fontsize =15) 

def get_cifar10():
    (X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
    X_train=X_train.reshape(X_train.shape[0],32,32,3).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test

def get_cifar10_batch(filename):                 
    """
    Load a single batch of CIFAR from the given file
    """
    with open(filename, 'rb') as f:
        dic = pickle.load(f , encoding='latin1')
        X = dic['data']                        # X : array 10000*3072= 0:1024 red channel 32*32,1024:2048 green channel
                                               #   ,2048:3072: blue channel
        Y = dic['labels']                       
        #X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('float64')   #for cnn
        Y = np.array(Y)
    return X, Y       

def plot_cifar10(idx):
    """
    important: 
    X: array 50000*3072
    """
    X,Y,_,_=get_cifar10()           # in get_cifar10 has been divided by 255.0
    X=X.reshape(50000,3,32,32)            # in cnn????????(50000,32,32,3)?????????????/
    
    path=r'data\cifar-10-batches-py\batches.meta'
    with open(path , 'rb') as f:
        dic=pickle.load(f,encoding="latin1")
        label_names=dic['label_names']
        
    red=X[idx][0]
    green=X[idx][1]
    blue=X[idx][2]
    image=np.dstack((red,green,blue))
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(image,interpolation='bicubic')
    ax.set_title('Category ={}'.format(label_names[Y[idx]]),fontsize =15)

        # Download the required dataset,split into data , labels

import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import sys
import gc
import json


def get_dataset(dataset,model):
        
    if dataset=='mnist':
        if model=='mlp':
            X_train,Y_train,X_test,Y_test=get_mnist_mlp()
            
        elif model in ('cnn1','cnn2','cnn3'):
            X_train ,Y_train,X_test,Y_test=get_mnist_cnn()
            
    elif dataset=='cifar10':
        X_train,Y_train,X_test,Y_test=get_cifar10()
                
    return X_train,Y_train,X_test,Y_test

def get_mnist_mlp():            
    (X_train,Y_train),(X_test,Y_test)=mnist.load_data()
    X_train=X_train.reshape(X_train.shape[0],28*28).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0],28*28).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test
    
def get_mnist_cnn():
    (X_train,Y_train),(X_test,Y_test)=mnist.load_data()
    X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test
    
def plot_mnist(idx):
    X,Y,_,_=get_mnist()                  # X is list,Y is array
    X=np.array(X).reshape(60000,28,28)
    image=X_train[idx]
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(X[idx],cmap=plt.cm.binary)        #interpolation='bicubic'
    ax.set_title('label ={}'.format(Y[idx]),fontsize =15) 

def get_cifar10():
    (X_train,Y_train),(X_test,Y_test)=cifar10.load_data()
    X_train=X_train.reshape(X_train.shape[0],32,32,3).astype('float32')
    X_train=X_train/255.0                  # [0,1]
    Y_train=to_categorical(Y_train,num_classes=10) 
    X_test=X_test.reshape(X_test.shape[0], 32, 32, 3).astype('float32')
    X_test=X_test/255.0      
    Y_test=to_categorical(Y_test,num_classes=10) 
    return X_train,Y_train,X_test,Y_test

def get_cifar10_batch(filename):                 
    """
    Load a single batch of CIFAR from the given file
    """
    with open(filename, 'rb') as f:
        dic = pickle.load(f , encoding='latin1')
        X = dic['data']                        # X : array 10000*3072= 0:1024 red channel 32*32,1024:2048 green channel
                                               #   ,2048:3072: blue channel
        Y = dic['labels']                       
        #X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype('float64')   #for cnn
        Y = np.array(Y)
    return X, Y       

def plot_cifar10(idx):
    """
    important: 
    X: array 50000*3072
    """
    X,Y,_,_=get_cifar10()           # in get_cifar10 has been divided by 255.0
    X=X.reshape(50000,3,32,32)            # in cnn????????(50000,32,32,3)?????????????/
    
    path=r'data\cifar-10-batches-py\batches.meta'
    with open(path , 'rb') as f:
        dic=pickle.load(f,encoding="latin1")
        label_names=dic['label_names']
        
    red=X[idx][0]
    green=X[idx][1]
    blue=X[idx][2]
    image=np.dstack((red,green,blue))
    fig = plt.figure(figsize=(3,3))
    ax = fig.add_subplot(111)
    ax.imshow(image,interpolation='bicubic')
    ax.set_title('Category ={}'.format(label_names[Y[idx]]),fontsize =15)
