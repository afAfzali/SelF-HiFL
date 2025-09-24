import numpy as np
import pickle
import tracemalloc
import random
import os
import psutil
import time
import shutil
os.environ['NUMEXPR_MAX_THREADS']='16'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import numexpr as ne
import time
import matplotlib.pyplot as plt
import sys
import gc
import ctypes
import tensorflow as tf
from tensorflow.keras.models import clone_model
from utils import compare
from utils import save_data
from utils import plot_image
from utils import check_target_client_existence
from utils import create_file
from utils import save_accuracy_changes_to_excel
from utils import show_confusion_matrix
from utils import distribution_dataset
from utils import load_model
from utils import calculate_label_frequencies
from utils import create_excel_attacks_test
from utils import define_train_data_for_attacks
from utils import test_attacks_on_clients
from utils import seconds_to_scientific
from utils import compare_attack_results
from utils import confusion_matrix_difference
from client import Client
from edgeserver import Edgeserver
from server import Server 
from attack import Attack
from datasets_partitioning.mnist_femnist import get_dataset
from datasets_partitioning.mnist_femnist import k_niid_equal_size_split
from datasets_partitioning.mnist_femnist import k_niid_equal_size_split_2
from datasets_partitioning.mnist_femnist import Gaussian_noise
from datasets_partitioning.mnist_femnist import get_classes
from datasets_partitioning.mnist_femnist import random_edges
from datasets_partitioning.mnist_femnist import iid_equal_size_split
from datasets_partitioning.mnist_femnist import iid_nequal_size_split
from datasets_partitioning.mnist_femnist import niid_labeldis_split
from datasets_partitioning.mnist_femnist import get_clients_femnist_cnn_with_reduce_writers_k_classes
from datasets_partitioning.mnist_femnist import get_clients_femnist_cnn_with_reduce_writers_k_classes_2
from tensorflow.keras.models import load_model
from model.initialize_model import create
from tensorflow.keras.utils import plot_model,to_categorical
from datetime import datetime
import jdatetime

# =============================================================================================================
#                                                Partitioning                
# =============================================================================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

dataset="mnist"                
if dataset=='cifar10':
    image_shape=(32,32,3)
    num_labels=10
    lr=0.005                             #0.01  for sgd                 lr=0.001 for adam
    model="cnn_federaser"   # cnn2
    batch_size=64

elif dataset=="mnist":
    image_shape=(28,28,1)
    num_labels=10
    lr=0.01
    model="cnn1"   #or cnn2, cnn3
    batch_size=32
    
elif dataset=='femnist':
    image_shape=(28,28,1)
    num_labels=15   # number classes of 62 classes   # 🔹
    train_size=21000
    test_size=9000 
    test_server_size=800
    model="cnn1"
    label_reduce=12
    batch_size=32
    lr=0.01

#folder=f"IID ({dataset})"   # non-IID
folder=f"non-IID ({dataset})" 
total_tr_size=1800
num_shadows=30  #30
attack_lr=0.001
attack_batch_size=32
attack_model="attack_xgboost"   # "attack_xgboost"  "attack_mlp"
if attack_model=="attack_xgboost":
    param_grid={'max_depth':[2,4,6],'learning_rate':[0.1,0.01,0.001],'n_estimators':[200,400,600,1000]}
attack_epochs=500  #500    for attack_mlp
global_round=10   #10             
epochs=10                          #  number of local update 
ul_epochs=5   #epochs    #5
community_delta=2
global_delta=2
community_round=6   #6             #  number of edge aggregation 
num_edges=2 
num_clients=4
fraction_clients=1             # fraction of participated clients
beta=0.5 
mean=0
loss="categorical_crossentropy"      #optimizer is "Adam"
metrics=["accuracy"]
verbose=1  
# seed=4   
# np.random.seed(seed)
# random.seed(seed)
optimizer=tf.keras.optimizers.SGD(learning_rate=lr)


tracemalloc.start()
process=psutil.Process()
start_rss=process.memory_info().rss

seed=205
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)
tf.config.experimental.enable_op_determinism()         # for GPU determinism (slower but ensures reproducibility)

#tf.keras.utils.set_random_seed(20)  # Global seed


# =============================================================================
#                              creating folders
# =============================================================================

now=datetime.now()
shamsi_date=jdatetime.datetime.fromgregorian(datetime=now)
folder_1=f"SelF-HiFL Results\SelF-HiFL ({folder})  {shamsi_date.strftime(f'%Y-%m-%d . . . %H-%M/figs')}"
folder_2=f"SelF-HiFL Results\SelF-HiFL ({folder})  {shamsi_date.strftime(f'%Y-%m-%d . . . %H-%M/clients/historical-models (clients)')}"
#folder_3=f"Results HiFU {shamsi_date.strftime(f'%Y-%m-%d . . . %H-%M/edges/{folder}/historical-models (edges)')}"
folder_4=f"SelF-HiFL Results\SelF-HiFL ({folder})  {shamsi_date.strftime(f'%Y-%m-%d . . . %H-%M/edges/historical-updates (clients)')}"
folder_5=f"SelF-HiFL Results\SelF-HiFL ({folder})  {shamsi_date.strftime(f'%Y-%m-%d . . . %H-%M/server/historical-models (edges)')}"
folder_6=f"SelF-HiFL Results\SelF-HiFL ({folder})  {shamsi_date.strftime(f'%Y-%m-%d . . . %H-%M/server/historical-models (server)')}"
folder_7=f"SelF-HiFL Results\SelF-HiFL ({folder})  {shamsi_date.strftime(f'%Y-%m-%d . . . %H-%M/MIA/shadows')}"
folder_8=f"SelF-HiFL Results\SelF-HiFL ({folder})  {shamsi_date.strftime(f'%Y-%m-%d . . . %H-%M/MIA/attacks')}"

os.makedirs(folder_1)
os.makedirs(folder_2)
#os.makedirs(folder_3)
os.makedirs(folder_4)
os.makedirs(folder_5)
os.makedirs(folder_6)
os.makedirs(folder_7)
os.makedirs(folder_8)

# ======================================================================================================
#                                             🟩    Shadows    🟩
# =======================================================================================================

path=folder_7.removesuffix('/shadows')
for sh_idx in range(num_shadows):
    if dataset=="mnist" or dataset=="cifar10":
        X_train,Y_train,X_test,Y_test=get_dataset(dataset,model) 
        #X_train,Y_train=X_train[total_tr_size:],Y_train[total_tr_size:]  # old
        X_train,Y_train=X_train[45000:],Y_train[45000:]    # new
        X_test,Y_test,X_test_server,Y_test_server=X_test[:8000],Y_test[:8000],X_test[9500:],Y_test[9500:] # server's test remains the same during learning

        train_idx=list(range(len(X_train)))
        indxs=np.random.choice(train_idx,total_tr_size,replace=False)
        X_train,Y_train=X_train[indxs],Y_train[indxs]
        save_data(folder_7,X_train,sh_idx,"Xtrain")
        save_data(folder_7,Y_train,sh_idx,"Ytrain")
        
        test_idx=list(range(len(X_test)))
        indxs=np.random.choice(test_idx,total_tr_size,replace=False)         # for shadows: train size = test size
        X_test,Y_test=X_test[indxs],Y_test[indxs]
        save_data(folder_7,X_test,sh_idx,"Xtest")
        save_data(folder_7,Y_test,sh_idx,"Ytest")
        
        #     ***********clients_iid*****************
        #print('\n** randomly are assigned clients to edgesevers **')
        clients=[]
        edges=[]
        
        train_partitions,test_partitions=iid_equal_size_split(X_train,Y_train,X_test,Y_test,num_clients)
          
        for i in range(num_clients):
            clients.append(Client(i,train_partitions[i],test_partitions[i],dataset,model,loss,metrics,
                                                             lr,batch_size,image_shape,num_labels)) 
        assigned_clients_list=random_edges(num_edges,num_clients) 
        for edgeid in range(num_edges):
            edges.append(Edgeserver(edgeid,assigned_clients_list[edgeid],dataset,model,loss,metrics,lr,image_shape,num_labels))
            for client_name in assigned_clients_list[edgeid]:               
                index=int(client_name.split('_')[1])-1               
                edges[edgeid].client_registering(clients[index])
        clients_per_edge=int(num_clients/num_edges)
        server=Server(dataset,model,loss,metrics,lr,image_shape,num_labels,X_test_server,Y_test_server)   
        
        print(tracemalloc.get_traced_memory()) 
        del X_train,Y_train,X_test,Y_test,train_partitions,test_partitions,assigned_clients_list
        gc.collect()
        print(tracemalloc.get_traced_memory()) 
        
    # assigning edges to server 
    for edge in edges:                                   
        server.edgeserver_registering(edge)         

    with open(fr"./{path}/shadow-results.txt","a") as sh_results:  
        print(f'\t\t\t\t** shadow-{sh_idx+1} **\n',file=sh_results)   
        for edge in edges:
            print(f'\t\t\t\t** {edge.name} **',file=sh_results)   
            cidxs=[]
            for client_name in edge.cnames:
                index=int(client_name.split('_')[1])-1
                cidxs.append(index)
            edge_clients=[clients[i] for i in cidxs]
            distribution_dataset(edge_clients,num_labels,"train",sh_results)
            distribution_dataset(edge_clients,num_labels,"test",sh_results)
        calculate_label_frequencies(edges,clients,sh_results)     # the clients of all edges 
        distribution_dataset(server,num_labels,"test",sh_results)


    # ------------------------------------------------------------------------------------------------------------------------
    #                        Hierarchical Federated Learning for shadow 
    # ------------------------------------------------------------------------------------------------------------------------
        #print(f"\n\t\t\tshadow-{sh_idx+1}:\n",file=sh_results)
        print("\n\t\t\tAccuracy of the global model on ... :\n\n",file=sh_results)
        print("\t\t\t\tinitial :\n",file=sh_results)
        print(f"the server's test data : {server.test(flag='save-return')}",file=sh_results)
    #server.model.save(fr"./{path}/my_model.h5") 

    server.save_model(0,folder_7,f"initial-shadow-{sh_idx+1}")   
    for global_r in range(1,global_round+1):    
        print(f'==================================== global round {global_r} starts now ================================================')
        server.refresh_server_buffer()
        for edge in edges:
            server.send_to_edgeserver(edge)                              
        for community_r in range(1,community_round+1):
            #print(f'--------------------------------------Community round {community_r} starts now ---------------------------------------') 
            for edge in edges:
                #print(f'************ {edge.name} starts ******************')
                edge.refresh_edgeserver_buffer()     
                for client_name in edge.cnames: 
                    #print(f"\n--------------------------------> {client_name} be selected:")
                    index=int(client_name.split('_')[1])-1
                    edge.send_to_client(clients[index])    
                    # print("Checking the equality of weights: ",compare(edge.model.get_weights(),clients[index].model.get_weights()))
                    
                    clients[index].local_model_train_and_update(epochs,verbose=0)  #,global_r,community_r,folder_2)
                    clients[index].send_to_edgeserver(edge)       # buffer
                    # print("Checking the equality of weights: ",compare(edge.buffer[client_name],clients[index].model_update.get_weights()))
                    # print(f"Checking the equality of sizes: ",edge.participated_sample[client_name]==clients[index].train_num)
                edge.aggregate()    # global_r,community_r
    
                #print(f'************ {edge.name} has ended ******************')
        #************end for/// iteration in edges
            print(f'--------------------------------------Community round {community_r} has ended ---------------------------------------')
        #*********** end for///edge aggregation        
        for edge in edges:                            
            edge.send_to_server(server)     # server' buffer
            # print("Checking the equality of weights: ",compare(edge.model.get_weights(),server.buffer[edge.name]))
        server.aggregate()   # global_r,community_r
        print(f'==================================== Global round {global_r} has ended ================================================')
    server.save_model(global_round,folder_7,f"final-shadow-{sh_idx+1}") 
    
    with open(fr"./{path}/shadow-results.txt", "a") as sh_results:  
        print(f"----------------------------------------------------------------------",file=sh_results)
        print("\t\t\t\tafter HiFL :\n ",file=sh_results)
        print(f"the server's test data : {server.test(flag='save-return')}",file=sh_results)
        print('\n*************************************************************************************************',file=sh_results)
        print('*************************************************************************************************',file=sh_results)

column_titles=["initial","after HiFL"]
row_titles=[]
data=[]
for sh_idx in range(num_shadows):                           
    row_titles.append(f"shadow_{sh_idx+1}") 
    server.load_model(0,folder_7,f"initial-shadow-{sh_idx+1}")
    initial_sh_acc=server.test()   # on server's test data     
    server.load_model(global_round,folder_7,f"final-shadow-{sh_idx+1}")
    learned_sh_acc=server.test()   
    data.append([initial_sh_acc,learned_sh_acc])
    
#path=folder_7.removesuffix('/shadows')
create_file(fr"./{path}/(shadows) acc on server's test.xlsx",row_titles,column_titles,data)    
## ========================================================================================================
#                             🟪🟪      The beginning of Learning - Unlearning - Retraining     🟪🟪
# ========================================================================================================

# -----------------------------------------------------------------------------
#                              partitioning and assigning
# -----------------------------------------------------------------------------

if dataset=="mnist" or dataset=="cifar10":
    X_train,Y_train,X_test,Y_test=get_dataset(dataset,model) 
    #X_train,Y_train=X_train[:total_tr_size],Y_train[:total_tr_size]  # old
    X_train,Y_train=X_train[:30000],Y_train[:30000]    # new
    train_idx=list(range(len(X_train)))
    indxs=np.random.choice(train_idx,total_tr_size,replace=False)
    X_train,Y_train=X_train[indxs],Y_train[indxs]
    X_test,Y_test,X_test_server,Y_test_server=X_test[8000:9500],Y_test[8000:9500],X_test[9500:],Y_test[9500:]
    
    print('1 : clients_iid (equal size)\n'
          '2 : clients_iid (nonequal size)\n'
          '3 : each client owns data samples of a fixed number of labels\n'
          '4 : each client(and edge) owns data samples of a different feature distribution\n'
          '5 : each client owns a proportion of the samples of each label\n')
    flag1=int(input('select a number:')) 
    #     ***********clients_iid*****************
    if flag1 in (1,2):  
        #print('\n** randomly are assigned clients to edgesevers **')
        folder=f"IID ({dataset})"
        clients=[]
        edges=[]

        if flag1==1:
            train_partitions,test_partitions=iid_equal_size_split(X_train,Y_train,X_test,Y_test,num_clients)
        else:
            train_partitions,test_partitions=iid_nequal_size_split(X_train,Y_train,X_test,Y_test,num_clients,beta)        
        for i in range(num_clients):
            clients.append(Client(i,train_partitions[i],test_partitions[i],dataset,model,loss,metrics,
                                                             lr,batch_size,image_shape,num_labels)) 
        assigned_clients_list=random_edges(num_edges,num_clients) 
        for edgeid in range(num_edges):
            edges.append(Edgeserver(edgeid,assigned_clients_list[edgeid],dataset,model,loss,metrics,lr,image_shape,num_labels))
            for client_name in assigned_clients_list[edgeid]:               
                index=int(client_name.split('_')[1])-1               
                edges[edgeid].client_registering(clients[index])
        clients_per_edge=int(num_clients/num_edges)
        server=Server(dataset,model,loss,metrics,lr,image_shape,num_labels,X_test_server,Y_test_server)   

        print(tracemalloc.get_traced_memory()) 
        del X_train,Y_train,X_test,Y_test,train_partitions,test_partitions,assigned_clients_list
        gc.collect()
        print(tracemalloc.get_traced_memory()) 

    #     ********** each edge owns data samples of a fixed number of labels ********** 
    elif flag1==3:  
        folder="non-IID-mnist"
        clients_per_edge=int(num_clients/num_edges)
        k1=int(input('\nk1 : number of labels for each edge  ?  '))
        k2=int(input('k2 : number of labels for clients per edge  ?  '))
        print(f'\n** assign each edge {clients_per_edge} clients with {k1} classes'
              f'\n** assign each client samples of {k2}  classes of {k1} edge classes')

        label_list=list(range(num_labels))
        X_train,Y_train,X_test,Y_test,party_labels_list=k_niid_equal_size_split_2(X_train,Y_train,X_test,
                                                                            Y_test,num_edges,label_list,k1,beta,flag1)  
        clients=[]
        edges=[]
        index=0  
        for edgeid in range(num_edges):           
            train_partitions,test_partitions=k_niid_equal_size_split_2(X_train[edgeid],Y_train[edgeid],X_test[edgeid],
                                                    Y_test[edgeid],clients_per_edge,party_labels_list[edgeid],k2,beta)
            assigned_clients=[]
            for i in range(clients_per_edge):
                clients.append(Client(index,train_partitions[i],test_partitions[i],dataset,model,loss,metrics,
                                                             lr,batch_size,image_shape,num_labels))   
                assigned_clients.append(index)
                index+=1
            assigned_clients=list(map(lambda x :f'client_{x+1}',assigned_clients))
            edges.append(Edgeserver(edgeid,assigned_clients,dataset,model,loss,metrics,lr,image_shape,num_labels))
            for client_name in assigned_clients:                 
                idx=int(client_name.split('_')[1])-1                
                edges[edgeid].client_registering(clients[idx])
        server=Server(dataset,model,loss,metrics,lr,image_shape,num_labels,X_test_server,Y_test_server)   
        print(tracemalloc.get_traced_memory()) 
        del X_train,X_test,Y_train,Y_test,test_partitions,train_partitions
        gc.collect()  
        print(tracemalloc.get_traced_memory()) 

    #     ********** each edge owns data samples of a different feature distribution ********** 
    #     ***** each edge owns data samples of 10 labels but each client owns data samples of one or 10 labels ***** 
    elif flag1==4:   
        folder=f"non-IID ({dataset})"
        original_std=float(input('\noriginal standard deviation for gaussian noise  ?  '))
        k=int(input('k : number of labels for clients of each edge  ?  '))  

        X_train,Y_train,X_test,Y_test=iid_equal_size_split(X_train,Y_train,X_test,Y_test,num_edges,flag1) 

        #basic_std=0.1      
        edges=[]
        clients=[]
        clients_per_edge=int(num_clients/num_edges)
        labels_list=list(range(num_labels)) 
        mean=0      
        index=0 
        for edgeid in range(num_edges):
            train_noisy_edge,test_noisy_edge=Gaussian_noise(X_train[edgeid],X_test[edgeid],original_std,edgeid,num_edges,mean)
            train_party_partitions,test_party_partitions=k_niid_equal_size_split(train_noisy_edge,Y_train[edgeid],test_noisy_edge, 
                                                                                 Y_test[edgeid],clients_per_edge,labels_list,k)
            assigned_clients=[]
            for i in range(clients_per_edge):
                clients.append(Client(index,train_party_partitions[i],test_party_partitions[i],dataset,model,loss,metrics,
                                                             lr,batch_size,image_shape,num_labels))  
                assigned_clients.append(index)
                index+=1
            assigned_clients=list(map(lambda x :f'client_{x+1}',assigned_clients))
            edges.append(Edgeserver(edgeid,assigned_clients,dataset,model,loss,metrics,lr,image_shape,num_labels))
            for client_name in assigned_clients:                  
                idx=int(client_name.split('_')[1])-1                
                edges[edgeid].client_registering(clients[idx])
        server=Server(dataset,model,loss,metrics,lr,image_shape,num_labels,X_test_server,Y_test_server)   
        print(tracemalloc.get_traced_memory()) 
        del X_train,Y_train,X_test,Y_test,train_noisy_edge,test_noisy_edge,train_party_partitions,test_party_partitions
        gc.collect()
        print(tracemalloc.get_traced_memory())

    #     ************** each client owns a proportion of the samples of each label **************
    elif flag1==5: 
        folder=f"non-IID ({dataset})"
        train_partitions,test_partitions=niid_labeldis_split(X_train,Y_train,X_test,Y_test,num_clients,beta)
        print("ok")
        clients=[]
        edges=[]
        clients_per_edge=int(num_clients/num_edges)
        index=0  
        for edgeid in range(num_edges):                           
            assigned_clients=[]
            for _ in range(clients_per_edge):
                # client_classes=get_classes(train_partitions[index])
                clients.append(Client(index,train_partitions[index],test_partitions[index],dataset,model,loss,metrics,
                                                             lr,batch_size,image_shape,num_labels))  
                assigned_clients.append(index)
                index+=1
            assigned_clients=list(map(lambda x :f'client_{x+1}',assigned_clients))
            edges.append(Edgeserver(edgeid,assigned_clients,dataset,model,loss,metrics,lr,image_shape,num_labels))
            for client_name in assigned_clients:                 
                idx=int(client_name.split('_')[1])-1               
                edges[edgeid].client_registering(clients[idx])
        server=Server(dataset,model,loss,metrics,lr,image_shape,num_labels,X_test_server,Y_test_server)   

        print(tracemalloc.get_traced_memory()) 
        del X_train,Y_train,X_test,Y_test,train_partitions,test_partitions
        gc.collect()
        print(tracemalloc.get_traced_memory()) 
    
    
elif dataset=="femnist":    # delete
    print('1 : equal size\n'
          '2 : equal size + reducing writers')
    flag1=int(input('select a number:'))
    folder=f"non-IID ({dataset})"
    """
    print("\nUsing a locally saved model?\n"
            "1 : YES\n"
            "0 : NO\n")
    replace=int(input('select a number:'))
    """
    if flag1==1:   
        print('\n** randomly are assigned clients to edgesevers **')
        train_partitions=equal_size_split(train_size,num_labels,num_clients,"train")
        print("here!")
        test_partitions=equal_size_split(test_size,num_labels,num_clients,"test")
        
    elif flag1==2:
        
        print('\n** randomly are assigned clients to edgesevers **')
        train_partitions,test_partitions,X_test_server,Y_test_server=get_clients_femnist_cnn_with_reduce_writers_k_classes_2(num_clients,train_size,
                                                                                               test_size,num_labels,label_reduce,test_server_size)
        print("partitinong ...end !")
    
    clients=[]
    edges=[]
    for i in range(num_clients):
        clients.append(Client(i,train_partitions[i],test_partitions[i],dataset,model,loss,metrics,
                                                             lr,batch_size,image_shape,num_labels))     
    assigned_clients_list=random_edges(num_edges,num_clients) 
    for edgeid in range(num_edges):
        edges.append(Edgeserver(edgeid,assigned_clients_list[edgeid],dataset,model,loss,metrics,lr,image_shape,num_labels))
        for client_name in assigned_clients_list[edgeid]:               
            index=int(client_name.split('_')[1])-1               
            edges[edgeid].client_registering(clients[index])
    clients_per_edge=int(num_clients/num_edges)
    server=Server(dataset,model,loss,metrics,lr,image_shape,num_labels,X_test_server,Y_test_server)  

    print(tracemalloc.get_traced_memory()) 
    del train_partitions,test_partitions,assigned_clients_list
    gc.collect()
    print(tracemalloc.get_traced_memory())

# assigning edges to server 
for edge in edges:                                   
    server.edgeserver_registering(edge)         

path=folder_1.removesuffix('/figs') 
with open(fr"./{path}/results.txt","w") as f_results:    
    print(f"dataset: {dataset}\nnum_labels: {num_labels}\nlr: {lr}",file=f_results)
    print(f"model: {model}\nbatch_size: {batch_size}\nglobal_round: {global_round}",file=f_results)
    print(f"l_epochs: {epochs}\nul_epochs: {ul_epochs}\ncommunity_round: {community_round}\nglobal_delta: {global_delta}",file=f_results)
    print(f"community_delta: {community_delta}\nnum_edges: {num_edges}\nnum_clients: {num_clients}\t\t{folder}\t\tflag1: {flag1}",file=f_results)
    print(f"num_shadows: {num_shadows}\nattack_lr: {attack_lr}\nattack_batch_size: {attack_batch_size}\nattack_model: {attack_model}\nattack_epochs: {attack_epochs}",file=f_results)
    print("============================================================================",file=f_results)
    
    for edge in edges:
        print(f'\t\t\t\t** {edge.name} **',file=f_results)   
        cidxs=[]
        for client_name in edge.cnames:
            index=int(client_name.split('_')[1])-1
            cidxs.append(index)
        edge_clients=[clients[i] for i in cidxs]
        distribution_dataset(edge_clients,num_labels,"train",f_results)
        distribution_dataset(edge_clients,num_labels,"test",f_results)
    calculate_label_frequencies(edges,clients,f_results)     # the clients of all edges 
    distribution_dataset(server,num_labels,"test",f_results)

# ===========================================================================================================================
#                      🟩        Step 1:    Hierarchical Federated Learning           🟩 
# ===========================================================================================================================
with open(fr"./{path}/results.txt", "a") as f_results:  
    print("\n\t\t\tAccuracy of the global model on ... :\n\n",file=f_results)
    print("\t\t\t\tinitial :\n",file=f_results)
    print(f"the server's test data : {server.test(flag='save-return')}",file=f_results)
#server.model.save(fr"./{path}/my_model.h5")
learning_time=[]
server.save_model(0,folder_6,"initial")   # for unlearning    1:  itr_1
start_time=time.time()
for global_r in range(1,global_round+1):    
    print(f'==================================== global round {global_r} starts now ================================================')
    server.refresh_server_buffer()
    for edge in edges:
        server.send_to_edgeserver(edge)                              
    for community_r in range(1,community_round+1):
        print(f'--------------------------------------Community round {community_r} starts now ---------------------------------------') 
        for edge in edges:
            print(f'************ {edge.name} starts ******************')
            edge.refresh_edgeserver_buffer()     
            for client_name in edge.cnames: 
                print(f"\n--------------------------------> {client_name} be selected:")
                index=int(client_name.split('_')[1])-1
                edge.send_to_client(clients[index])    
                # print("Checking the equality of weights: ",compare(edge.model.get_weights(),clients[index].model.get_weights()))
                
                clients[index].local_model_train_and_update(epochs,verbose=0)  #,global_r,community_r,folder_2)
                clients[index].send_to_edgeserver(edge)       # buffer
                # print("Checking the equality of weights: ",compare(edge.buffer[client_name],clients[index].model_update.get_weights()))
                # print(f"Checking the equality of sizes: ",edge.participated_sample[client_name]==clients[index].train_num)
                
            if (global_r%global_delta==1 or global_r==global_round) and (community_r%community_delta==1 or community_r==community_round):   #❌
                edge.aggregate(global_r,community_r,folder_4,"saving update")
            else:
                edge.aggregate()    # global_r,community_r

            print(f'************ {edge.name} has ended ******************')
    #************end for/// iteration in edges
        print(f'--------------------------------------Community round {community_r} has ended ---------------------------------------')
    #*********** end for///edge aggregation        
    for edge in edges:                            
        edge.send_to_server(server)     # server' buffer
        # print("Checking the equality of weights: ",compare(edge.model.get_weights(),server.buffer[edge.name]))
    if global_r==1 and community_r==community_round:                #❌💣    
        server.aggregate(global_r,community_r,folder_5,"saving update")
    else:
        server.aggregate()   # global_r,community_r
    print(f'==================================== Global round {global_r} has ended ================================================')
    
learning_time.append(seconds_to_scientific(time.time()-start_time))    

server.save_model(global_r,folder_6,"learned") 

with open(fr"./{path}/results.txt", "a") as f_results:  
    print(f"----------------------------------------------------------------------",file=f_results)
    print("\t\t\t\tafter HiFL :\n ",file=f_results)
    print(f"the server's test data : {server.test(flag='save-return')}",file=f_results)

# print(process.memory_info().rss-start_rss)
# print(tracemalloc.get_traced_memory())
# tracemalloc.stop()

# ==========================================================================================================================
#                    🟩        Step 2:    Hierarchical Federated Unlearning              🟩 
# ==========================================================================================================================
 
ta=int(input(f'select a target client from {num_clients} clients:  ')) 
target_client=f"client_{ta}"
target_edge=check_target_client_existence(edges,target_client)

with open(fr"./{path}/results.txt", "a") as f_results:  
    print(f"the target client's train data : {server.test(clients[ta-1],1)}",file=f_results)
    
for client in clients:
    server.test(client)

# with open(fr"./{path}/summary.txt", "w") as f_client: 
#     print("after hifl :",file=f_client)
#     for client in clients:
#         print(client.name,":" ,server.test(client,1),file=f_client)

for edge in edges:
    edge.refresh_edgeserver_size()
    for client_name in edge.calibrating_cnames:
        index=int(client_name.split('_')[1])-1
        edge.client_registering(clients[index])   
target_edge_index=int(target_edge.split('_')[1])-1
server.refresh_server_size() 
for edge in edges:                                   
    server.edgeserver_registering(edge) 
server.load_model(0,folder_6,"initial")    # first unlearned global model   

global_ul_rounds=list(range(1,global_round+1,global_delta))
if global_round not in global_ul_rounds: 
    global_ul_rounds.append(global_round)
community_ul_rounds=list(range(1,community_round+1,community_delta))
if community_round not in community_ul_rounds: 
    community_ul_rounds.append(community_round)
    
start_time=time.time()
for global_r in global_ul_rounds:    
    print(f'==================================== Global round {global_r} starts now ================================================')
    server.refresh_server_buffer()          
    for edge in edges:
        server.send_to_edgeserver(edge)                              
    for community_r in community_ul_rounds:
        print(f'--------------------------------------Community round {community_r} starts now ---------------------------------------') 
        for edge in edges:
            if edge.name==target_edge:
                print(f'************ {edge.name} (target edge) starts ******************')
                edge.refresh_edgeserver_buffer()   
                for client_name in edge.calibrating_cnames:
                    print(f"\n--------------------------------> {client_name} be selected:")
                    index=int(client_name.split('_')[1])-1
                    edge.send_to_client(clients[index]) 
                    # print("Checking the equality of weights: ",compare(edge.model.get_weights(),clients[index].model.get_weights()))

                    clients[index].local_model_train_and_update(ul_epochs,verbose=0) # ,global_r,community_r,folder_2  
                    clients[index].send_to_edgeserver(edge)       # buffer
                    # print("Checking the equality of weights: ",compare(edge.buffer[client_name],clients[index].model_update.get_weights()))
                    # print(f"Checking the equality of sizes: ",edge.participated_sample[client_name]==clients[index].train_num)
                
                edge.calibrate_and_aggregate_weights(global_r,community_r,folder_4)
                print(f'************ {edge.name} (target edge) has ended ******************')
            else:
                if global_r==1 and community_r!=community_round:
                    continue
                elif global_r==1 and community_r==community_round:
                    server.load_edge_model(edge.name,global_r,community_r,folder_5)
                    print(f'************ {edge.name} model is loaded ******************')
                else:
                    print(f'************ {edge.name} starts ******************')
                    edge.refresh_edgeserver_buffer()   
                    for client_name in edge.cnames:
                        print(f"\n--------------------------------> {client_name} be selected:")
                        index=int(client_name.split('_')[1])-1
                        edge.send_to_client(clients[index])    
                        clients[index].local_model_train_and_update(ul_epochs,verbose)  # ,global_r,community_r,folder_2
                        clients[index].send_to_edgeserver(edge)       # buffer
                    edge.calibrate_and_aggregate_weights(global_r,community_r,folder_4)
                    print(f'************ {edge.name} has ended ******************')
        print(f'--------------------------------------Community round {community_r} has ended ---------------------------------------') 
    if global_r==1:
        edges[target_edge_index].send_to_server(server)
    else:
        for edge in edges:
            edge.send_to_server(server)
    server.aggregate()
    _=server.test(flag='save-return')  
    
    print(f'==================================== Global round {global_r} has ended ================================================')
learning_time.append(seconds_to_scientific(time.time()-start_time))     

with open(fr"./{path}/results.txt", "a") as f_o:  
    print(f"----------------------------------------------------------------------",file=f_o)
    print("\t\t\t\tafter SelF-HiFL :\n ",file=f_o)
    print(f"the server's test data : {server.test_acc[-1]}",file=f_o)
    print(f"the target client's train data : {server.test(clients[ta-1],1)}",file=f_o)
    
for client in clients:
    server.test(client)
    
# with open(fr"./{path}/summary.txt", "a") as f_client: 
#     print("\nafter SelF-HiFL :",file=f_client)
#     for client in clients:
#         print(client.name,":" ,server.test(client,1),file=f_client)
    
server.save_model(global_r,folder_6,"unlearned") 

# print(process.memory_info().rss-start_rss)
# print(tracemalloc.get_traced_memory())
# tracemalloc.stop()

# =============================================================================================================================
#                     🟩          Step 3:     Retraining         🟩 
# ==============================================================================================================================
    
server.load_model(0,folder_6,"initial")    

start_time=time.time()
for global_r in range(1,global_round+1):    
    print(f'==================================== global round {global_r} starts now ================================================')
    server.refresh_server_buffer()
    for edge in edges:
        server.send_to_edgeserver(edge)                              
              
    for community_r in range(1,community_round+1):
        print(f'--------------------------------------Community round {community_r} starts now ---------------------------------------') 
        for edge in edges:
            print(f'************ {edge.name} starts ******************')
            edge.refresh_edgeserver_buffer()     
            for client_name in edge.calibrating_cnames: 
                print(f"\n--------------------------------> {client_name} be selected:")
                index=int(client_name.split('_')[1])-1
                edge.send_to_client(clients[index])    
                # print("Checking the equality of weights: ",compare(edge.model.get_weights(),clients[index].model.get_weights()))
                
                clients[index].local_model_train_and_update(epochs,verbose)  #,global_r,community_r,folder_2)
                clients[index].send_to_edgeserver(edge)       # buffer
                # print("Checking the equality of weights: ",compare(edge.buffer[client_name],clients[index].model_update.get_weights()))
                # print(f"Checking the equality of sizes: ",edge.participated_sample[client_name]==clients[index].train_num)
                
            edge.aggregate()    

            print(f'************ {edge.name} has ended ******************')
    #************end for/// iteration in edges
        print(f'--------------------------------------Community round {community_r} has ended ---------------------------------------')
    #*********** end for///edge aggregation        
    for edge in edges:                            
        edge.send_to_server(server)     # server' buffer
        # print("Checking the equality of weights: ",compare(edge.model.get_weights(),server.buffer[edge.name]))
        
    server.aggregate()   # global_r,community_r
    print(f'==================================== Global round {global_r} has ended ================================================')
learning_time.append(seconds_to_scientific(time.time()-start_time))
server.save_model(global_r,folder_6,"retrained")

with open(fr"./{path}/results.txt", "a") as f_o:  
    print(f"----------------------------------------------------------------------",file=f_o)
    print("\t\t\t\tafter Retraining :\n: ",file=f_o)
    print(f"the server's test data : {server.test(flag='save-return')}",file=f_o)
    print(f"the target client's train data : {server.test(clients[ta-1],1)}",file=f_o)

for client in clients:
    server.test(client)              # in tain_acc of each client = (0 : hifl)- ( 1: self ) - (2 : retrain )
    
# with open(fr"./{path}/summary.txt", "a") as f_client: 
#     print("\nafter retrain :",file=f_client)
#     for client in clients:
#         print(client.name,":", server.test(client,1),file=f_client)
    
# =============================================================================
#          Reporting training accuracy for clients at various steps
# =============================================================================
column_titles=["after HiFL","after SelF-HiFL","after Retrain"]
row_titles=[]
data=[]
for i in range(num_clients):
    row_titles.append(f"client_{i+1}") 
    data.append(clients[i].train_acc)
    
path=folder_1.removesuffix('/figs')
create_file(fr"./{path}/train accuracy of clients.xlsx",row_titles,column_titles,data,target_client,num_clients)
save_accuracy_changes_to_excel(fr"./{path}/train accuracy of clients.xlsx",target_client,num_clients)

#-------------
row_titles=["on server's test data","on target's train data"," ","during the time (seconds)"," "," "," ","test acc during unlearning"]
data=[] 
indices=[1,-2,-1]             #  1: learned , -1: retrained , -2:unleraned 
#data.append(server.test_acc[1:])     # index:  0: initial   , 1: learned , -1: retrained , -2:unleraned ,  rest: unlearned in g_rounds
data.append([server.test_acc[i] for i in indices])
data.append(clients[ta-1].train_acc)    # ta-1: index of target client 
data.append([])
data.append(learning_time)
for _ in range(3):
    data.append([])
data.append([server.test_acc[i] for i in [0]+list(range(2,len(server.test_acc)-1))])    

create_file(fr"./{path}/accu on target train, server test.xlsx",row_titles,column_titles,data)

server.load_model(global_round,folder_6,"learned")
learned_diagonal_values=show_confusion_matrix(fr"./{path}/accu on target train, server test.xlsx","learned",server,num_labels)

server.load_model(global_round,folder_6,"unlearned")
unlearned_diagonal_values=show_confusion_matrix(fr"./{path}/accu on target train, server test.xlsx","unlearned",server,num_labels)

server.load_model(global_round,folder_6,"retrained")
retrained_diagonal_values=show_confusion_matrix(fr"./{path}/accu on target train, server test.xlsx","retrained",server,num_labels)

confusion_matrix_difference(fr"./{path}/accu on target train, server test.xlsx",learned_diagonal_values,unlearned_diagonal_values,
                                                  retrained_diagonal_values,clients,num_labels,target_client)
# ========================================================================================================
#                                🟪🟪    The End of Learning - Unlearning - Retraining    🟪🟪
# ========================================================================================================
# ========================================================================================================
#                                🟪🟪   The beginning of the Membership Inference Attack    🟪🟪
# ========================================================================================================

# -------------------------------------------------------------------------------------------------------
#                            🟩    defining attack models via shadow models    🟩
# ------------------------------------------------------------------------------------------------------

temp_model=clone_model(server.model)     

train_X_attack,train_Y_attack=define_train_data_for_attacks(temp_model,global_round,folder_7,num_shadows,num_labels,attack_model)

attacks=[]    
for at_i in range(num_labels):
    if attack_model=="attack_mlp":   # بهتر بنویسم ارگومان ها رو
        attacks.append(Attack(at_i,train_X_attack[at_i],train_Y_attack[at_i],attack_lr,num_labels,attack_model,attack_batch_size))
    elif attack_model=="attack_xgboost":
        attacks.append(Attack(at_i,train_X_attack[at_i],train_Y_attack[at_i],attack_lr,num_labels,attack_model,attack_batch_size,param_grid))

    attacks[at_i].local_train(attack_model,attack_epochs,verbose)   # بهتر بنویسم ارگومان ها رو 
    attacks[at_i].save_model(at_i,"final",folder_8,attack_model)  

path=folder_7.removesuffix('/shadows')
learned_confusion_results,learned_metrics_results,learned_attack_results=test_attacks_on_clients(temp_model,attacks,clients,global_round,folder_6,
                                                                                "learned",num_labels,path,target_client,attack_model)
retrained_confusion_results,retrained_metrics_results,retrained_attack_results=test_attacks_on_clients(temp_model,attacks,clients,global_round,folder_6,
                                                                                "retrained",num_labels,path,target_client,attack_model)
unlearned_confusion_results,unlearned_metrics_results,unlearned_attack_results=test_attacks_on_clients(temp_model,attacks,clients,global_round,folder_6,
                                                                                 "unlearned",num_labels,path,target_client,attack_model)

data=[learned_confusion_results,retrained_confusion_results,unlearned_confusion_results]
create_excel_attacks_test(num_clients,num_labels,"confusion-matrix",["tn","fp","fn","tp"],data,target_client,fr"./{path}/(attacks' acc) on clients' train.xlsx")
data=[learned_metrics_results,retrained_metrics_results,unlearned_metrics_results]
create_excel_attacks_test(num_clients,num_labels,"metrics",["accuracy","precision","recall","f1_score"],data,target_client,fr"./{path}/(attacks' acc) on clients' train.xlsx")

data={}
for key in learned_attack_results:
    true_l=learned_attack_results[key][0]  
    learn_attack_predict=learned_attack_results[key][1] 
    retrain_attack_predict=retrained_attack_results[key][1]  
    unlearn_attack_predict=unlearned_attack_results[key][1] 
    
    combined=[[t_l,l_a,rt_a,ul_a] for t_l,l_a,rt_a,ul_a, in zip(true_l,learn_attack_predict,retrain_attack_predict,unlearn_attack_predict)]
    data[key]=combined  
    
compare_attack_results(data,fr"./{path}/(attacks' acc) on clients' train.xlsx")
    
