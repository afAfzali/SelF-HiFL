# intialization
###training for one iteration and test function

from model.mlp import SimpleMLP
from model.cnn import Mnist_CNN_1
from model.cnn import Mnist_CNN_2
from model.cnn import Mnist_CNN_3
from model.cnn import Mnist_CNN_federaser
from model.cnn import Cifar10_CNN_federaser
from model.cnn import Cifar10_CNN_2

def create(dataset,model,loss,metrics,lr,image_shape,num_labels):  
    
    #if dataset=="mnist":
     #   if model=="mlp":
      #      m=SimpleMLP(784,10,loss,metrics,lr)
            
    if dataset=="mnist" or dataset=="femnist":
        if model=='cnn1':
            m=Mnist_CNN_1(loss,metrics,lr,image_shape,num_labels)
        elif model=='cnn2':
            m=Mnist_CNN_2(loss,metrics,lr,image_shape,num_labels)
        elif model=='cnn3':
            m=Mnist_CNN_3(loss,metrics,lr,image_shape,num_labels)
        elif model=="cnn_federaser":
            m=Mnist_CNN_federaser(loss,metrics,lr,image_shape,num_labels)
            
    elif dataset=="cifar10":
        if model=="cnn_federaser":
            m=Cifar10_CNN_federaser(loss,metrics,lr,image_shape,num_labels)
        elif model=="cnn2":
            m=Cifar10_CNN_2(loss,metrics,lr,image_shape,num_labels)
    return m
