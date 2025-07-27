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

def get_clients_femnist_cnn_with_reduce_writers_k_classes(num_parties,train_size,test_size,number_classes):    # ok
    num_train_samples_party=int(train_size/num_parties)
    num_test_samples_party=int(test_size/num_parties)
    train_test_ratio=num_train_samples_party/num_test_samples_party
    train_partitions=[0]*num_parties
    test_partitions=[0]*num_parties
    for i in range(num_parties):
        num_test_samples_party=int(test_size/num_parties)
        print(i,":")
        with open(fr'D:\python-projects(jupyter)\Per-Hie-GAN-2\LEAF\train\all_data_{i}_niid_0_keep_0_train_9.json') as f_in:
            r1=json.load(f_in)  
        with open(fr'D:\python-projects(jupyter)\Per-Hie-GAN-2\LEAF\test\all_data_{i}_niid_0_keep_0_test_9.json') as f_in:
            r2=json.load(f_in) 
        X_train=[]
        Y_train=[]
        X_test=[]
        Y_test=[]
        for (_,v1),(_,v2) in zip(r1["user_data"].items() ,r2["user_data"].items()):
            if num_test_samples_party!=0:
                l1=[0]*number_classes
                l2=[0]*number_classes
                zero_indices = []
                idxs=[]
                for j1 in v1["y"]:
                    if j1 in list(range(number_classes)):
                        l1[j1]+=1
                for idx, value in enumerate(l1):
                    if value == 0:
                        zero_indices.append(idx)
                for j2 in v2["y"]:
                    if j2 in list(range(number_classes)):
                        l2[j2]+=1
                for idx, value in enumerate(l2):
                    if value == 0:
                        zero_indices.append(idx)
                rest_labels=list(set(list(range(number_classes)))-set(zero_indices))
                train_labels_frequency=[0]*len(rest_labels)
                test_labels_frequency=[0]*len(rest_labels)
                for idx in range(len(rest_labels)):
                    train_labels_frequency[idx]=l1[rest_labels[idx]]
                    test_labels_frequency[idx]=l2[rest_labels[idx]]
                selected_labels=[]
                for idx in range(len(rest_labels)):
                    if num_test_samples_party-test_labels_frequency[idx]>=0:
                        if train_labels_frequency[idx]>=int(train_test_ratio*test_labels_frequency[idx]):
                            selected_labels.append(rest_labels[idx])
                            num_test_samples_party-=test_labels_frequency[idx]
                        else:
                            selected_labels.append(-1)
                    else:
                        break 
                train_idxs=[]
                for label in selected_labels:
                    if label!=-1:
                        idx_l=[]
                        for j,d in enumerate(v1["y"]):
                            if label==d:
                                idx_l.append(j)
                        selected_idxs=np.random.choice(idx_l,int(test_labels_frequency[selected_labels.index(label)]*train_test_ratio),
                                                                   replace=False)
                        train_idxs.extend(selected_idxs)
                X_train.extend(np.array(v1["x"])[train_idxs])        
                Y_train.extend(np.array(v1["y"])[train_idxs])

                test_idxs=[]
                for label in selected_labels:
                    if label!=-1:
                        idx_l=[]
                        for j,d in enumerate(v2["y"]):
                            if label==d:
                                idx_l.append(j)
                        selected_idxs=np.random.choice(idx_l,test_labels_frequency[selected_labels.index(label)],replace=False)
                        test_idxs.extend(selected_idxs)
                X_test.extend(np.array(v2["x"])[test_idxs])
                Y_test.extend(np.array(v2["y"])[test_idxs])
            else:
                break

        X_train=np.array(X_train)
        X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
        Y_train=np.array(Y_train)
        Y_train=to_categorical(Y_train,number_classes)
        X_test=np.array(X_test)
        X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
        Y_test=np.array(Y_test)
        Y_test=to_categorical(Y_test,number_classes)
        train_partitions[i]=tf.data.Dataset.from_tensor_slices((X_train,Y_train))
        test_partitions[i]=tf.data.Dataset.from_tensor_slices((X_test,Y_test))
        del X_train,Y_train,X_test,Y_test
        gc.collect()
        
    return train_partitions,test_partitions

def get_clients_femnist_cnn_with_reduce_writers_k_classes_2(num_parties,train_size,test_size,number_classes,label_reduce):  
    num_train_samples_party=int(train_size/num_parties)
    num_test_samples_party=int(test_size/num_parties)
    train_test_ratio=num_train_samples_party/num_test_samples_party
    train_partitions=[0]*num_parties
    test_partitions=[0]*num_parties
    for i in range(num_parties):
        num_test_samples_party=int(test_size/num_parties)
        print(i,":")
        with open(fr'.\LEAF\train\all_data_{i}_niid_0_keep_0_train_9.json') as f_in:
            r1=json.load(f_in)  
        with open(fr'.\LEAF\test\all_data_{i}_niid_0_keep_0_test_9.json') as f_in:
            r2=json.load(f_in) 
        X_train=[]
        Y_train=[]
        X_test=[]
        Y_test=[]

        new_classes=list(np.random.choice(range(number_classes),label_reduce,replace=False))
        for (_,v1),(_,v2) in zip(r1["user_data"].items() ,r2["user_data"].items()):
            if num_test_samples_party!=0:
                l1=[0]*number_classes
                l2=[0]*number_classes
                zero_indices = []
                idxs=[]
                for j1 in v1["y"]:
                    if j1 in new_classes:
                        l1[j1]+=1
                for idx, value in enumerate(l1):
                    if value == 0:
                        zero_indices.append(idx)
                for j2 in v2["y"]:
                    if j2 in new_classes:
                        l2[j2]+=1
                for idx, value in enumerate(l2):
                    if value == 0:
                        zero_indices.append(idx)   
                rest_labels=list(set(new_classes)-set(zero_indices))
                train_labels_frequency=[0]*len(rest_labels)
                test_labels_frequency=[0]*len(rest_labels)
                for idx in range(len(rest_labels)):
                    train_labels_frequency[idx]=l1[rest_labels[idx]]
                    test_labels_frequency[idx]=l2[rest_labels[idx]]
                selected_labels=[]
                for idx in range(len(rest_labels)):
                    if num_test_samples_party-test_labels_frequency[idx]>=0:
                        if train_labels_frequency[idx]>=int(train_test_ratio*test_labels_frequency[idx]):
                            selected_labels.append(rest_labels[idx])
                            num_test_samples_party-=test_labels_frequency[idx]
                        else:
                            selected_labels.append(-1)
                    else:
                        break 
                train_idxs=[]
                for label in selected_labels:
                    if label!=-1:
                        idx_l=[]
                        for j,d in enumerate(v1["y"]):
                            if label==d:
                                idx_l.append(j)
                        selected_idxs=np.random.choice(idx_l,int(test_labels_frequency[selected_labels.index(label)]*train_test_ratio),
                                                                   replace=False)
                        train_idxs.extend(selected_idxs)
                X_train.extend(np.array(v1["x"])[train_idxs])        
                Y_train.extend(np.array(v1["y"])[train_idxs])
                test_idxs=[]
                for label in selected_labels:
                    if label!=-1:
                        idx_l=[]
                        for j,d in enumerate(v2["y"]):
                            if label==d:
                                idx_l.append(j)

                        selected_idxs=np.random.choice(idx_l,test_labels_frequency[selected_labels.index(label)],replace=False)
                        test_idxs.extend(selected_idxs)
                X_test.extend(np.array(v2["x"])[test_idxs])
                Y_test.extend(np.array(v2["y"])[test_idxs])

            else:
                break

        X_train=np.array(X_train)
        X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
        Y_train=np.array(Y_train)
        Y_train=to_categorical(Y_train,number_classes)
        X_test=np.array(X_test)
        X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
        Y_test=np.array(Y_test)
        Y_test=to_categorical(Y_test,number_classes)

        test_i=list(range(len(X_test)))
        server_i=np.random.choice(test_i,test_server_size_party,replace=False)
        test_i=list(set(test_i)-set(server_i)) 
        X_test_server.extend(X_test[server_i])
        Y_test_server.extend(Y_test[server_i])
            
        train_partitions[i]=tf.data.Dataset.from_tensor_slices((X_train,Y_train))
        test_partitions[i]=tf.data.Dataset.from_tensor_slices((X_test[test_i],Y_test[test_i]))
        del X_train,Y_train,X_test,Y_test
        gc.collect()
        
    return train_partitions,test_partitions


def iid_equal_size_split(train_data,train_label,test_data,test_label,num_parties,flag=None):  
    train_size=int(len(train_data)/num_parties)
    train_idx=list(range(len(train_data)))
    test_size=int(len(test_data)/num_parties)             
    test_idx=list(range(len(test_data)))
    if flag==None:
        train_partitions=[0]*num_parties
        test_partitions=[0]*num_parties
        for i in range(num_parties):
            indxs=np.random.choice(train_idx,train_size,replace=False)
            train_partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[indxs],train_label[indxs]))
            train_idx=list(set(train_idx)-set(indxs)) 
        for i in range(num_parties):
            indxs=np.random.choice(test_idx,test_size,replace=False)
            test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[indxs],test_label[indxs]))
            test_idx=list(set(test_idx)-set(indxs))                    
        return train_partitions,test_partitions
    else:
        train_data_partitions=[0]*num_parties
        train_label_partitions=[0]*num_parties
        test_data_partitions=[0]*num_parties
        test_label_partitions=[0]*num_parties
        for i in range(num_parties):
            indxs=np.random.choice(train_idx,train_size,replace=False)
            train_data_partitions[i]=train_data[indxs]
            train_label_partitions[i]=train_label[indxs]
            train_idx=list(set(train_idx)-set(indxs))  
        for i in range(num_parties):
            indxs=np.random.choice(test_idx,test_size,replace=False)
            test_data_partitions[i]=test_data[indxs]
            test_label_partitions[i]=test_label[indxs]
            test_idx=list(set(test_idx)-set(indxs))
        return train_data_partitions,train_label_partitions,test_data_partitions,test_label_partitions

"""         quantity skew         """                   
def iid_nequal_size_split(train_data,train_label,test_data,test_label,num_parties,beta=0.9):                    
    train_num_samples=len(train_data)
    test_num_samples=len(test_data)
    train_partitions=[0]*num_parties
    min_size_of_parties=0
    while min_size_of_parties<50:                  
        p=np.random.dirichlet(np.repeat(beta,num_parties))            
        size_parties=np.random.multinomial(train_num_samples, p)       
        min_size_of_parties=np.min(size_parties)
    train_idx=list(range(len(train_data)))
    for i,size in enumerate(size_parties):
        indxs=np.random.choice(train_idx,size,replace=False)
        train_partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[indxs],train_label[indxs]))
        train_idx=list(set(train_idx)-set(indxs))
    test_size=int(test_num_samples/num_parties)
    test_idx=list(range(len(test_data)))
    test_partitions=[0]*num_parties
    for i in range(num_parties):
        indxs=np.random.choice(test_idx,test_size,replace=False)
        test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[indxs],test_label[indxs]))
        test_idx=list(set(test_idx)-set(indxs)) 
    return train_partitions,test_partitions

"""         label distribution skew -->  distribution-based label imbalanced         """
def niid_labeldis_split(train_data,train_label,test_data,test_label,num_clients,beta):       

    # each client has a proportion of the samples of each label(Dirichlet distribution)
    # The size of the local data set is not equal
    
    num_labels=10 
    train_num_samples=len(train_data)
    train_i=np.array([np.argmax(train_label[idx]) for idx in range(len(train_label))])
    train_partitions=[0]*num_clients
    train_partitions_idxs=[[] for _ in range(num_clients)]
    for k in range(num_labels):
        k_idx=np.where(train_i==k)[0]
        np.random.shuffle(k_idx)
        min_size_of_labels=0
        while min_size_of_labels<10:    #<10:
            p=np.random.dirichlet(np.repeat(beta,num_clients))
            p=np.random.multinomial(len(k_idx),p)
            min_size_of_labels=np.min(p)
        for i,size in enumerate(p):
            idxs=np.random.choice(k_idx,size,replace=False)
            train_partitions_idxs[i].extend(idxs)
            k_idx=list(set(k_idx)-set(idxs))
    for i in range(num_clients):
            train_partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[train_partitions_idxs[i]],
                                                                        train_label[train_partitions_idxs[i]]))
    test_size=int(len(test_data)/num_clients)
    test_i=list(range(len(test_data)))
    test_partitions=[0]*num_clients
    for i in range(num_clients):
        idxs=np.random.choice(test_i,test_size,replace=False)
        test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[idxs],test_label[idxs]))
        test_i=list(set(test_i)-set(idxs)) 
    return train_partitions,test_partitions

"""         label distribution skew -->  quantity-based label imbalanced   """  # ok    به طور مساوی تفسیم میشه    
def k_niid_equal_size_split(train_data,train_label,test_data,test_label,num_parties,labels_list,k,flag=None): 
    
    """ k: number of lables for each party """
    # label_lists: ممکن است فقط چند تا برچسب مختلف از کل برچسب ها را داشته باشه به دلیل پارتیشن بندی مرحله دوم روی کلاینت ها
    # برای تقسیم بندی مرحله دوم،اندیس هاش میشه اندیس متناظر با لیبل که متفاوته با لیبل
    
    labels_index=np.arange(len(labels_list))
    times=[0]*len(labels_list) 
    party_labels_list=[] 
    z=0
    #if num_parties<num_labels:
    for i in range(num_parties):
        c=[]
        if z==0:
            idxs=np.random.choice(labels_index,k,replace=False)
            for idx in idxs:
                c.append(labels_list[idx])
                times[idx]+=1
            if len(np.where(np.array(times)==0)[0])>0:
                zero_list=list(np.where(np.array(times)==0)[0]) 
                z=1
        else:
            if len(zero_list)<k:
                for idx in zero_list:        
                    c.append(labels_list[idx])
                    times[idx]+=1
                rest_labels_list=list(set(labels_index)-set(zero_list))
                idxs=np.random.choice(rest_labels_list,k-len(zero_list),replace=False)
                for idx in idxs:
                    c.append(labels_list[idx])
                    times[idx]+=1
                z=0
            else:
                idxs=np.random.choice(zero_list,k,replace=False)
                for idx in idxs:
                    c.append(labels_list[idx])
                    times[idx]+=1
                zero_list=list(np.where(np.array(times)==0)[0]) 
                z=1
        party_labels_list.append(c)
    #print("train_len:",len(train_data))
    #print("len:",len(train_label))
    #print("test_len:",len(test_data))
    #print("len:",len(test_label))
    train_i=[np.argmax(train_label[idx]) for idx in range(len(train_label))]
    test_i=[np.argmax(test_label[idx]) for idx in range(len(test_label))]

    train_partition_idxs=[[] for _ in range(num_parties)]
    test_partition_idxs=[[] for _ in range(num_parties)]
    train_idx_l=[]
    test_idx_l=[]
    #print("train_i:",train_i)
    print("times:",times)
    for i,l in enumerate(labels_list):
        for j,d in enumerate(train_i):
            if d==l:
                train_idx_l.append(j)
        for j,d in enumerate(test_i):
            if d==l:
                test_idx_l.append(j)         
        np.random.shuffle(train_idx_l)
        np.random.shuffle(test_idx_l)
        train_split=np.array_split(train_idx_l,times[i])
        test_split=np.array_split(test_idx_l,times[i])
        #print("train_split:",train_split)
        index=0
        for j in range(num_parties):
            if l in party_labels_list[j]:
                train_partition_idxs[j].extend(train_split[index])
                test_partition_idxs[j].extend(test_split[index])            
                index+=1
        train_idx_l.clear()
        test_idx_l.clear()
    if flag==None:
        train_partitions=[0]*num_parties
        test_partitions=[0]*num_parties
        for i in range(num_parties): 
            train_partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[train_partition_idxs[i]],
                                                                    train_label[train_partition_idxs[i]]))
            test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[test_partition_idxs[i]],
                                                                    test_label[test_partition_idxs[i]]))
        return train_partitions,test_partitions
    else:
        tr_data=[0]*num_parties
        tr_label=[0]*num_parties
        te_data=[0]*num_parties
        te_label=[0]*num_parties
        for i in range(num_parties):
            tr_data[i]=train_data[train_partition_idxs[i]]
            tr_label[i]=train_label[train_partition_idxs[i]]
            te_data[i]=test_data[test_partition_idxs[i]]
            te_label[i]=test_label[test_partition_idxs[i]]
        return tr_data,tr_label,te_data,te_label,party_labels_list

"""         label distribution skew -->  quantity-based label imbalanced   """  # ok    توزیع دیریکله  
def k_niid_equal_size_split_2(train_data,train_label,test_data,test_label,num_parties,labels_list,k,beta,flag=None): 
    
    """ k: number of lables for each party """
    
    
    labels_index=np.arange(len(labels_list))
    times=[0]*len(labels_list) 
    party_labels_list=[] 
    z=0
    #if num_parties<num_labels:
    for i in range(num_parties):
        c=[]
        if z==0:
            idxs=np.random.choice(labels_index,k,replace=False)
            for idx in idxs:
                c.append(labels_list[idx])
                times[idx]+=1
            if len(np.where(np.array(times)==0)[0])>0:
                zero_list=list(np.where(np.array(times)==0)[0]) 
                z=1
        else:
            if len(zero_list)<k:
                for idx in zero_list:          # تابع رو برای حالات ممکن که همه لیبل ها در پارتیشن ها باشند نوشتم 
                    c.append(labels_list[idx])
                    times[idx]+=1
                rest_labels_list=list(set(labels_index)-set(zero_list))
                idxs=np.random.choice(rest_labels_list,k-len(zero_list),replace=False)
                for idx in idxs:
                    c.append(labels_list[idx])
                    times[idx]+=1
                z=0
            else:
                idxs=np.random.choice(zero_list,k,replace=False)
                for idx in idxs:
                    c.append(labels_list[idx])
                    times[idx]+=1
                zero_list=list(np.where(np.array(times)==0)[0]) 
                z=1
        party_labels_list.append(c)
    #print("train_len:",len(train_data))
    #print("len:",len(train_label))
    #print("test_len:",len(test_data))
    #print("len:",len(test_label))
    train_i=[np.argmax(train_label[idx]) for idx in range(len(train_label))]
    test_i=[np.argmax(test_label[idx]) for idx in range(len(test_label))]

    train_partition_idxs=[[] for _ in range(num_parties)]
    test_partition_idxs=[[] for _ in range(num_parties)]
    train_idx_l=[]
    test_idx_l=[]
    #print("train_i:",train_i)
    #print("times:",times)
    for i,l in enumerate(labels_list):
        for j,d in enumerate(train_i):
            if d==l:
                train_idx_l.append(j)
        for j,d in enumerate(test_i):
            if d==l:
                test_idx_l.append(j)         
        np.random.shuffle(train_idx_l)
        np.random.shuffle(test_idx_l)
        if times[i]==1:
            train_split=np.array_split(train_idx_l,times[i])
            test_split=np.array_split(test_idx_l,times[i])
        else:
            train_split=[[] for _ in range(times[i])]
            min_size_of_labels=0
            while min_size_of_labels==0:
                p=np.random.dirichlet(np.repeat(beta,num_parties))
                p=np.random.multinomial(len(train_idx_l),p)
                min_size_of_labels=np.min(p)
            #print("p",p)
            for k,size in enumerate(p):
                idxs=np.random.choice(train_idx_l,size,replace=False)
                train_split[k].extend(idxs)
                train_idx_l=list(set(train_idx_l)-set(idxs))
            test_split=np.array_split(test_idx_l,times[i])
        #print("train_split:",train_split)
        #print("train_idx_l",len(train_idx_l))
        #print("test_idx_l",len(test_idx_l))
        index=0
        for j in range(num_parties):
            if l in party_labels_list[j]:
                train_partition_idxs[j].extend(train_split[index])
                test_partition_idxs[j].extend(test_split[index])            
                index+=1
        train_idx_l.clear()
        test_idx_l.clear()
    if flag==None:
        train_partitions=[0]*num_parties
        test_partitions=[0]*num_parties
        for i in range(num_parties): 
            train_partitions[i]=tf.data.Dataset.from_tensor_slices((train_data[train_partition_idxs[i]],
                                                                    train_label[train_partition_idxs[i]]))
            test_partitions[i]=tf.data.Dataset.from_tensor_slices((test_data[test_partition_idxs[i]],
                                                                    test_label[test_partition_idxs[i]]))
        return train_partitions,test_partitions
    else:
        tr_data=[0]*num_parties
        tr_label=[0]*num_parties
        te_data=[0]*num_parties
        te_label=[0]*num_parties
        for i in range(num_parties):
            tr_data[i]=train_data[train_partition_idxs[i]]
            tr_label[i]=train_label[train_partition_idxs[i]]
            te_data[i]=test_data[test_partition_idxs[i]]
            te_label[i]=test_label[test_partition_idxs[i]]
        return tr_data,tr_label,te_data,te_label,party_labels_list


def random_edges(num_edges,num_clients):
    #randomly select clientsfor assign clients to edgesever 
    clients_per_edge=int(num_clients/num_edges)
    c_indxs=list(range(num_clients))
    assigned_clients=[]
    for edgeid in range(num_edges):
        assigned_c=np.random.choice(c_indxs,clients_per_edge,replace=False)
        c_indxs=list(set(c_indxs)-set(assigned_c))
        assigned_c=list(map(lambda x: f"client_{x+1}" ,assigned_c))
        assigned_clients.append(assigned_c)
    return assigned_clients

def get_classes(data_label):
    l=[0]*10
    for _,i in data_label:
        l[np.argmax(i)] += 1
    return list(np.where(np.array(l)!=0)[0])     

"""         feature distribution skew --->> noise_based feature imbalanced         """
def Gaussian_noise(train_data,test_data,original_std,idx,num_parties,mean):
    """
    for party idx :std = original_std*(idx/num_parties)
    image data and noisy_image_data must be scaled in [0, 1] 
    """
    std=original_std*idx/num_parties #  
    noisy_train_list=[]
    noisy_test_list=[]
    noise=np.random.randn(*train_data[0].shape)*std+mean
    for i in range(len(train_data)):
        #noise=np.random.randn(*train_data[i].shape)*std+mean
        train_noisy_data=np.clip(noise+train_data[i],0,1)
        noisy_train_list.append(train_noisy_data)
    for i in range(len(test_data)):
        #noise=np.random.randn(*train_data[i].shape)*std+mean
        test_noisy_data=np.clip(noise+test_data[i],0,1)
        noisy_test_list.append(test_noisy_data)
    return np.array(noisy_train_list),np.array(noisy_test_list)
