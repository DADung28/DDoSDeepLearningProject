# Import library for load data
import argparse
import ray
ray.init(num_cpus=24)
import modin.pandas as pd
#import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
import csv
# Import deeplearning library
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms 
from torch.utils.data import Dataset
import time
import os
import numpy as np 
from CustomFunction import *
from model_define import *

#----------------PARSE PARAMETER-----------------
parser = argparse.ArgumentParser(description='This program will take input file as CICFLowMeter flow information (csv file) and output trained model in [trained_model] training plot data [csv_plot]')
# parser.add_argument: Add parameter to program
WORK_PATH = os.getcwd()
parser.add_argument('--device', type = int, help='Define GPU device for training', default=0) # Required parameter
parser.add_argument('--dataset', help=f'Training dataset path (default: {WORK_PATH}/dataset_ddos/CICIDS2017.csv)', default=WORK_PATH + '/dataset_ddos/CICIDS2017.csv') # Required parameter
parser.add_argument('--model', help='Traning model selection (ANN, CNN, DNN) (default: CNN)', default='CNN') 
parser.add_argument('--ver', help='Traning model version name (Scheduled, NonScheduled, Test) (default: Test)', default='Test') 
parser.add_argument('--epoch_start', type = int, help='Epoch is start from here (default: 0)', default= 0) 
parser.add_argument('--scheduler', help='Add scheduler with this flag (default: False)', action='store_true') 
parser.add_argument('--lr', type = float, help='Training learning rate (default: 0.001)', default=0.01) 
parser.add_argument('--bs', type = int, help='Batch size for traning (Default: 10000)', default= 1000)
parser.add_argument('--type', help='Type of dataset ("CICIDS2019","CICIDS2019_anomaly","CICIDS2019_test","CICIDS2018","CICIDS2018_anomaly","CICIDS2017","CICIDS2017_anomaly" default: CICIDS2017_anomaly)', default='CICIDS2017_anomaly') 
args = parser.parse_args()

#----------------Define variable based of args----------
GPU_NUM = args.device
dataset_PATH = args.dataset
model_type = args.model  
ver = args.ver
epoch_start = args.epoch_start
scheduler_on = args.scheduler
learning_rate = args.lr
batch_size = args.bs
train_type = args.type
trained_model_dir_PATH = WORK_PATH + '/trained_model' + '/' + train_type + '/' + model_type + '_' + ver 
# Make folder for trained model
os.makedirs(trained_model_dir_PATH, exist_ok=True)
# Make folder for plot data file
trained_model_PATH = trained_model_dir_PATH + '/' + str(epoch_start + 1) + 'epoch.pth'
csv_plot_PATH = WORK_PATH + '/trained_model' + '/' + train_type + '/'  + model_type + '_' + ver + '.csv'
print('-------------PARAMETERS-------------')
print('Current work path: ',WORK_PATH)
print('Dataset PATH:', dataset_PATH)
print('Dataset type:', type)
print('Training on GPU: ',GPU_NUM)
print('Deep learning model: {}, ver: {}, epoch from: {}, learning rate: {}, batch_size: {}, scheduler: {}'.format(model_type, ver, epoch_start, learning_rate, batch_size, scheduler_on))
print('Trained model is saved at:', trained_model_PATH)
print('Plot data is saved at:', csv_plot_PATH)

# Define CUDA device 
device = torch.device(GPU_NUM if torch.cuda.is_available() else 'cpu')

print('-------------ANALYZE DATASET-------------')
analyze_start =  time.time()
#Load dataset
print("Loading dataset, please wait .....")
train = pd.read_csv(dataset_PATH)
#test = pd.read_csv('/home/jun/CICIDS2019/TestCIC2019.csv')
print("DONE!")
print("-------------------------")

# Fix columns name
print("Start fixing columns name")
train = fix_columns_name(train)
print("DONE!")
print("-------------------------")


# Drop NaN and inf
print("Start dropping NaN and Infinity contained row")
train = drop_NaN(train)
print("DONE!")
print("-------------------------")

# Encoding train dataset label
print("Start encoding flow labels to number")
train = encoding_labels(train_type, train)
print("DONE!")
print("-------------------------")
output_size = len(set(train['Label']))
# Drop num numberic columns ('Flow ID','Source IP','Destination IP', 'Timestamp', 'SimillarHTTP')
print("Dropping ('Unnamed: 0', 'Flow ID','Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'SimillarHTTP')")
try:
    train = drop_non_numberic_columns(train)
except:
    print("No columns to drop")
print("DONE!")
print("-------------------------")
input_size = len(train.columns)-1
output_size = len(set(train['Label']))
# Change data to numpy
print("Change data into numpy")
train = train.to_numpy()
train = np.asfarray(train)
print("DONE!")
print("-------------------------")

# Split dataset to data and label
print("Split numpy data to data and label")
print(train.shape)
print(input_size)
train_day_X = train[:, :input_size]
train_day_y = train[:, input_size]
print("DONE!")
print("-------------------------")

# Normalize data to range (0, 1)
print("Normalizing data into range [0,1] for training")
scaler = MinMaxScaler()
scaler.fit(train_day_X)
train_day_X = scaler.transform(train_day_X)
print("DONE!")
print("-------------------------")

# Slit training day data to train (80%), val (0.0002) and test (0.1998)
print("Split train, validate and test dataset: 80:10:10")
X_train, X_val_test, y_train, y_val_test = train_test_split(train_day_X, train_day_y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42) 
print("DONE!")
print('Train shape: {}, Validate shape: {}, Test shape: {}'.format(X_train.shape, X_val.shape, X_test.shape))
print("-------------------------")
analyze_end =  time.time()
print('Analyze data took: {:.2f}s'.format(analyze_end - analyze_start))

print('-------------TRAINING-------------')
Train_dataset = MyDataset(X_train.astype(np.float32), y_train.astype(np.float32))
Val_dataset = MyDataset(X_val.astype(np.float32), y_val.astype(np.float32))
Test_dataset = MyDataset(X_test.astype(np.float32), y_test.astype(np.float32))

# Create a data loader for batch training
train_loader = torch.utils.data.DataLoader(Train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(Val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=batch_size, shuffle=False)

# Define Model  
if model_type == 'ANN':
    model = ANN(input_size, output_size)
elif model_type == 'ANND':
    model = ANND(input_size, output_size)
elif model_type == 'DNN':
    model = DNN(input_size, output_size)
elif model_type == 'DNND':
    model = DNND(input_size, output_size)
elif model_type == 'CNN':
    model = CNN(input_size, output_size)
elif model_type == 'CNN2':
    model = CNN2(input_size, output_size)
elif model_type == 'CNN3':
    model = CNN3(input_size, output_size)
try:
    model.load_state_dict(torch.load(trained_model_dir_PATH + '/' + str(epoch_start) + 'epoch.pth'))
    print('Loaded trained model: {}'.format(trained_model_dir_PATH + '/' + str(epoch_start) + 'epoch.pth'))
except:
    print('No loaded trained model, train from scratch')
    pass
# Push model to GPU
model.to(device)
print(model)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Create csv file to plot data

f = open('./trained_model' + '/' + train_type + '/'  + model_type + '_' + ver + '.csv', 'w')
writer = csv.writer(f)
header = [str(batch_size) + 'batch', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
writer.writerow(header)
f.close()
# This function will calculate loss and accuracy on test
def get_test_loss_acc(net, criterion, data_loader):
    """A simple function that iterates over `data_loader` to calculate the overall loss"""
    testing_loss = []
    testing_acc = []
    avg_loss = 0
    avg_acc = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            # get the data to GPU (if available)
            inputs, labels = inputs.to(device), labels.to(device, dtype = torch.long) # Change to data type long to not rise error
            outputs = net(inputs)
            # calculate the loss for this batch
            loss = criterion(outputs, labels)
            # add the loss of this batch to the list
            testing_loss.append(loss.item())
            _, predicted = torch.max(outputs, 1) # Get prediction on batch (shape: batch_size)
            batch_correct_predicted = (predicted == labels).sum().item() # Get number of true prediction on batch
            testing_acc.append(batch_correct_predicted/batch_size*100) # Append prediction accuracy of this batch
    avg_loss = np.mean(testing_loss) 
    avg_acc = np.mean(testing_acc) 
    
    # calculate the average loss
    return {'loss': avg_loss, 'acc': avg_acc}

# Training loop
running_loss = []
running_acc = []
n_total_step = len(train_loader)
num_epochs = 25
for epoch in range(epoch_start, num_epochs):
    f = open('./trained_model' + '/' + train_type + '/'  + model_type + '_' + ver + '.csv', 'a')
    writer = csv.writer(f)
    # Total loss on all training sample
    total_loss = []
    # Total accuracy on all training sample
    total_acc = []
    epoch_start_time = time.time()
    for i, (data, labels) in enumerate(train_loader):
        # Batch size: 10000 
        # 1000000, 81
        labels = labels.to(device, dtype = torch.long) # Change labels to long data type for cross entropy
        data = data.to(device)
        
        # forward 
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Batch loss
        running_loss.append(loss.item())

        # Batch accuracy calculate
        _, predicted = torch.max(outputs, 1) # Get prediction on batch (shape: batch_size)
        batch_correct_predicted = (predicted == labels).sum().item() # Get number of true prediction on batch
        running_acc.append(batch_correct_predicted/batch_size*100) # Append prediction accuracy 
        if i % 10 == 0:
            # Calculate average train loss on 10000 batch
            avg_train_loss = np.mean(running_loss)
            running_loss.clear()
            # Calculate average train accuracy on 10000 batch
            avg_train_acc = np.mean(running_acc)
            running_acc.clear()
            # Adding batch loss to total loss
            
            total_loss.append(avg_train_loss)
            # Adding batch accuracy to total accuracy
            total_acc.append(avg_train_acc)
            
            epoch_end_time = time.time() 
            print(f'epoch {epoch + 1}/{num_epochs} ({epoch_end_time - epoch_start_time:.2f}s), step [{i+1}/{n_total_step}], train_loss = {avg_train_loss:.3f}, train_acc = {avg_train_acc:.3f}')
            
    if scheduler_on:
        scheduler.step()
    # Calculate average test loss on all val dataset
    avg_test_loss = get_test_loss_acc(model, criterion, val_loader)['loss']
    # Calculate average train accuracy on all val dataset
    avg_test_acc = get_test_loss_acc(model, criterion, val_loader)['acc']
    # Write plot data to csv file
    plot_data = [epoch + 1,np.mean(total_loss),np.mean(total_acc),avg_test_loss,avg_test_acc]
    writer.writerow(plot_data)
    f.close()   
    print('Epoch: [{}/{}] ({:.2f}s), train_loss: {:.3f}, train_accuracy: {:.3f}, test_loss = {:.3f}, test_acc = {:.3f}'.format(epoch + 1, num_epochs, epoch_end_time - epoch_start_time, np.mean(total_loss), np.mean(total_acc), avg_test_loss, avg_test_acc))
    saved_model_PATH = './trained_model' + '/' + '/' + train_type + '/' + model_type + '_' + ver + '/' + str(epoch + 1) + 'epoch.pth'
    torch.save(model.state_dict(), saved_model_PATH)
