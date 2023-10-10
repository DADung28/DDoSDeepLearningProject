# Import library for load data
import argparse
import ray
ray.init(num_cpus=28)
import modin.pandas as pd
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
import seaborn as sns
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from category_encoders import OrdinalEncoder
from numpy import genfromtxt
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
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np 
from CustomFunction import *
from model_define import *
WORK_PATH = os.getcwd()
validated_dir = WORK_PATH  + '/validated_balanced'
os.makedirs(validated_dir, exist_ok=True) 
#----------------PARSE PARAMETER-----------------
parser = argparse.ArgumentParser(description='This program will take trained model and validate on dataset]')
# parser.add_argument: Add parameter to program
WORK_PATH = os.getcwd()
parser.add_argument('--device', type = int, help='Define GPU device for training', default=0) # Required parameter
parser.add_argument('--ver', help='Traning model version name (Scheduled, NonScheduled, Test) (default: Scheduled)', default='Scheduled') 
parser.add_argument('--dataset', help=f'Training dataset path (default: {WORK_PATH}/dataset/TrainCIC2019.csv)', default=WORK_PATH + '/dataset/TrainCIC2019.csv') # Required parameter
parser.add_argument('--model', help='Trained model name (ANN, ANND, DNN, DNND)(default: ANN)', default='ANN') 
parser.add_argument('--epoch', help='Trained model epoch_number (default: 100)', default='100') 
parser.add_argument('--bs', type = int, help='Batch size for traning (Default: 10000)', default= 1000)
parser.add_argument('--trained_type', help='Type of trained model ("train" for TrainCIC2019.csv or "test" for TestCIC2019.csv, default: train)', default='train') 
parser.add_argument('--test_type', help='Type of dataset ("train" for TrainCIC2019.csv or "test" for TestCIC2019.csv, default: train)', default='train') 
args = parser.parse_args()

#----------------Define variable based of args----------
GPU_NUM = args.device
dataset_PATH = args.dataset
model_type = args.model
ver = args.ver
epoch_num = args.epoch
batch_size = args.bs
test_type = args.test_type
trained_type = args.trained_type
trained_model_PATH =  WORK_PATH + '/trained_model' + '/' + trained_type + '/' + model_type + '_' + ver + '/' + epoch_num + 'epoch.pth'
# Make folder for validate data file
print('-------------PARAMETERS-------------')
print('Current work path:',WORK_PATH)
print('Dataset PATH:', dataset_PATH)
print('Dataset type:', test_type)
print('Training on GPU:',GPU_NUM)
print('Batch size:', batch_size)
print('Trained model PATH:', trained_model_PATH)

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
train = encoding_labels(test_type, train)
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

# Slit training day data to train (90%), val (5%) and test (5%)
print("Split train, validate and test dataset: 80:10:10")
X_train, X_val_test, y_train, y_val_test = train_test_split(train_day_X, train_day_y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42) 
print("DONE!")
print('Train shape: {}, Validate shape: {}, Test shape: {}'.format(X_train.shape, X_val.shape, X_test.shape))
print("-------------------------")
analyze_end =  time.time()
print('Analyze data took: {:.2f}s'.format(analyze_end - analyze_start))

print("Balancing test dataset into 5000:5000")
test_dataset = np.concatenate((X_test,y_test.reshape(y_test.shape[0],1)),axis=1)
new_test_dataset = []
count_0 = 0
count_1 = 0
for flow in test_dataset:
    if flow[77] == 0 and count_0 < 5000:
        count_0 += 1
        new_test_dataset.append(flow) 
    if flow[77] == 1 and count_1 < 5000:
        count_1 += 1
        new_test_dataset.append(flow) 
new_test_dataset = np.array(new_test_dataset)
X_test = new_test_dataset[:, :input_size]
y_test = new_test_dataset[:, input_size]
print("DONE!")
print("-------------------------")

print('-------------Validating-------------')
Test_dataset = MyDataset(X_test.astype(np.float32), y_test.astype(np.float32))
# Create a data loader for batch training
test_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=batch_size, shuffle=False)
examples = iter(test_loader)
samples, labels = next(examples)
print(samples.shape, labels.shape)
print(samples[0].dtype)

# Create Model to load
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

model.load_state_dict(torch.load(trained_model_PATH, map_location=device))
# Push model to GPU
model.to(device)
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
print(model)

def get_test_loss_acc(net, criterion, data_loader):
    """A simple function that iterates over `data_loader` to calculate the overall loss"""
    testing_loss = []
    testing_acc = []
    avg_loss = 0
    avg_acc = 0
    # Define confusion matrix
    c = torch.zeros((output_size, output_size))
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            # Get the data to GPU (if available)
            inputs, labels = inputs.to(device), labels.to(device, dtype = torch.long) # Change to data type long to not rise error
            outputs = net(inputs)
            # Calculate the loss for this batch
            loss = criterion(outputs, labels)
            # Add the loss of this batch to the list
            testing_loss.append(loss.item())
            _, predicted = torch.max(outputs, 1) # Get prediction on batch (shape: batch_size)
            # Add up confusion matrix in each batch to confusion matrix
            c += ConfusionMatrix(labels, predicted, output_size)
            batch_correct_predicted = (predicted == labels).sum().item() # Get number of true prediction on batch
            testing_acc.append(batch_correct_predicted/batch_size*100) # Append prediction accuracy of this batch
    # Calculate the average loss and accuracy
    avg_loss = np.mean(testing_loss) 
    avg_acc = np.mean(testing_acc) 
    return {'loss': avg_loss, 'acc': avg_acc, 'confusion_matrix':c}

result = get_test_loss_acc(model, criterion, test_loader)

avg_test_loss = result['loss']
avg_test_acc = result['acc']
c_matrix = result['confusion_matrix'] 
validate_result = Validate(c_matrix)
actual = validate_result['actual']
total_acc = validate_result['accuracy'] 
precision = validate_result['precision'] 
precision_dict = {}
for i,p in enumerate(precision):
    precision_dict.update({i:np.around(float(p),decimals=3)})
    
recall = validate_result['recall'] 
recall_dict = {}
for i,r in enumerate(recall):
    recall_dict.update({i:np.around(float(r),decimals=3)})
f1_score = validate_result['f1_score'] 
f1_score_dict = {}
for i,f in enumerate(f1_score):
    f1_score_dict.update({i:np.around(float(f),decimals=3)})
txt_file_PATH = validated_dir + '/' + trained_type.split('_')[0] + '_' + test_type.split('_')[0] + '.txt' 
f = open(txt_file_PATH, 'w')

f.write('Test average accuracy: {:.2f}%, Test average loss: {:.3f}'.format(avg_test_acc, avg_test_loss))
f.write('\n')
print('Test average accuracy: {:.2f}%, Test average loss: {:.3f}'.format(avg_test_acc, avg_test_loss))
f.write(f'Confusion matrix:\n {c_matrix.type(torch.int).numpy()}')
f.write('\n')
print('Confusion matrix:\n', c_matrix.type(torch.int).numpy())
f.write(f'Benign : DDoS = {int(actual[0])} : {int(actual[1])} = {(actual[0]/torch.sum(actual)*100).item():.2f} : {(actual[1]/torch.sum(actual)*100).item():.2f}')
f.write('\n')
print(f'Benign : DDoS = {int(actual[0])} : {int(actual[1])} = {(actual[0]/torch.sum(actual)*100).item():.2f} : {(actual[1]/torch.sum(actual)*100).item():.2f}')
f.write(f'Total accuracy: {total_acc*100:.2f}%')
f.write('\n')
print(f'Total accuracy: {total_acc*100:.2f}%')
f.write(f'Precision: {precision_dict}')
f.write('\n')
print(f'Precision: {precision_dict}')
f.write(f'Recall: {recall_dict}')
f.write('\n')
print(f'Recall: {recall_dict}')
f.write(f'f1_score: {f1_score_dict}')
f.write('\n')
print(f'f1_score: {f1_score_dict}')
f.close()