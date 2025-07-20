import numpy as np 
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Alignment,PatternFill,Border,Side
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter

def average_weights(w,sample_num):
    """
    w: list of lists
    sample_num: list of dataset sizes
    return: list
    """
    total_sample_num=sum(sample_num)
    a=[]
    avg_w=[]
    for i in range(len(w)):
        t=[]
        for j in range(len(w[0])):
            t.append(w[i][j]*(sample_num[i]/total_sample_num))
        a.append(t) 
    avg_w=[sum(k) for k in zip(*a)]  
    return avg_w

def sum_list(a,b):
    result=[]
    for x,y in zip(a,b):
        if isinstance(x,list) and isinstance(y,list):
            result.append(sum_lists(x,y))  
        else:
            result.append(x+y) 
    return result

def l2_norm(x):
    flat=flatten(x)
    return sum(x**2 for x in flat)**0.5

def flatten(x):
    flat_list=np.concatenate([a.flatten() for a in x])
    return flat_list

def multiply(ratio,a):
    result=[]
    for x in a:
        if isinstance(x,list):
            result.append(multiply(ratio,a))
        else:
            result.append(ratio*x)
    return result

def compare(a,b):
    for x,y in zip(a,b):
        if not np.array_equal(x,y):
            return False
    return True
        
def check_target_client_existence(edges,client_name):
    for edge in edges:
        if client_name in edge.cnames:
            target_edge=edge.name
            edge.calibrating_cnames=list(set(edge.cnames)-set([client_name]))
        else: 
            edge.calibrating_cnames=edge.cnames
    print(f"{client_name} belongs to {target_edge}")
    return target_edge

def distribution_dataset(data_structure,num_labels,flag,file_name):
    if isinstance(data_structure,list):
        if flag=="train":
            print("\t\t\t\ttrain:\n",file=file_name)
            for client in data_structure:
                n_labels=[0]*num_labels
                for i,j in client.x:
                    n_labels[np.argmax(j.numpy())]+=1
                print("--------------------",file=file_name)
                print( client.name,":",n_labels,"-->",np.sum(n_labels),file=file_name)      
            print('\n-----------------------------------------------------------------',file=file_name)
        else:
            print("\n\t\t\t\ttest:\n",file=file_name)
            for client in data_structure:
                n_labels=[0]*num_labels
                for i,j in client.y:
                    n_labels[np.argmax(j.numpy())]+=1
                print("--------------------",file=file_name)
                print( client.name,":",n_labels,"-->",np.sum(n_labels),file=file_name)
            print('\n=================================================================',file=file_name)
    else:
        print("\n\t\t\t\tserver test:\n",file=file_name)
        n_labels=[0]*num_labels
        for i,j in data_structure.y:
            n_labels[np.argmax(j.numpy())]+=1
        print(" server:",n_labels,"-->",np.sum(n_labels),file=file_name) 
        print('\n=================================================================',file=file_name)
              
def plot_image(idx,X,Y,dataset):
    image=X[idx]
    plt.figure(figsize=(5,5))
    if dataset=="mnist":
        plt.imshow(image,cmap='gray_r')   
        plt.title(f'label={np.argmax(Y[idx])}',fontsize=10)    
    elif dataset=="cifar10":
        labels=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        plt.imshow(image,interpolation='lanczos')   
        plt.title(f'label={labels[np.argmax(Y_train[idx])]}',fontsize=10) 

def create_file(filename,row_titles,column_titles,data,target_client=None,num_clients=None,delimiter=','):
    wb=Workbook()
    ws=wb.active
    ws.append([""]+column_titles)

    for i,(row_title,row_data) in enumerate(zip(row_titles,data)):
        ws.append([row_title]+row_data)
    
    if target_client:
        red_fill=PatternFill(start_color='FFC7CE',end_color='FFC7CE',fill_type='solid')
        rule=FormulaRule(formula=[f'$A2="{target_client}"'], fill=red_fill, stopIfTrue=True)
        last_col=len(column_titles)+1  
        ws.conditional_formatting.add(f"A2:{get_column_letter(last_col)}{num_clients+1}", rule)
    
    for column in ws.columns:      #auto-adjust column widths
        max_length=0
        column_letter=get_column_letter(column[0].column)
        for cell in column:
            try:
                if len(str(cell.value))>max_length:
                    max_length=len(str(cell.value))
            except:
                pass
        
        adjusted_width=(max_length+2)* 1.2
        ws.column_dimensions[column_letter].width=adjusted_width

    for row in ws.iter_rows():
        for cell in row:
            if cell.value:
                cell.alignment=Alignment(horizontal='center',vertical='center')
    wb.save(filename)
    
def save_accuracy_changes_to_excel(filename,target_client,num_clients):
    wb=load_workbook(filename)
    ws=wb.active
    ws['G1']="Avg of clients' acc except target client"
    column_letter='G'
    max_length=len(ws['G1'].value)
    ws.column_dimensions[column_letter].width=max_length+1 
    ws['H1']="target client's acc differences"
    column_letter='H'
    max_length=len(ws['H1'].value)
    ws.column_dimensions[column_letter].width=max_length+1
    ws['F2']="SelF-HiFL acc - HiFL acc"
    ws['F3']="Retrain acc - HiFL acc"
    column_letter='F'
    max_length=len(ws['F3'].value)
    ws.column_dimensions[column_letter].width=max_length+1 
    
    data=[]
    target_index=int(target_client.split('_')[1]) 
    row_list=list(set(list(range(2,num_clients+2)))-set([target_index+1]))  
    
    for row in row_list: 
        cell1=round(ws.cell(row=row,column=3).value-ws.cell(row=row,column=2).value,2)  # SelF-HiFL acc - Hifl acc
        cell2=round(ws.cell(row=row,column=4).value-ws.cell(row=row,column=2).value,2)  # retrain acc - Hifl acc
        data.append([cell1,cell2])
        
    ws['G2'],ws['G3']=np.round(np.mean(data,axis=0),3)
    ws['H2']=round(ws.cell(row=target_index+1,column=3).value-ws.cell(row=row,column=2).value,3)   # SelF-HiFL acc - Hifl acc
    ws['H3']=round(ws.cell(row=target_index+1,column=4).value-ws.cell(row=row,column=2).value,3)   # retrain acc - Hifl acc
    
    for row in ws.iter_rows():
        for cell in row:
            if cell.value:  # Only align cells with content
                cell.alignment=Alignment(horizontal='center',vertical='center')
    
    # blue_fill=PatternFill(start_color="00B0F0", end_color="00B0F0", fill_type="solid")
    # for row in [1,2,3]:         
    #     for col in [6,7,8]:      
    #         ws.cell(row=row, column=col).fill=blue_fill
    thin_border=Border(left=Side(style='thin'),right=Side(style='thin'),
        top=Side(style='thin'),bottom=Side(style='thin'))
    
    for row in [1,2,3]:        
        for col in [6,7,8]:      
            ws.cell(row=row, column=col).border = thin_border
    wb.save(filename)
