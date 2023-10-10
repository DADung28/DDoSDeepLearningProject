# Import library for load data
import ray
ray.init(num_cpus=8)
import modin.pandas as pd
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

# Define CUDA device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('-------------ANALYZE DATASET-------------')
#Load dataset
print("Loading dataset, please wait .....")
train = pd.read_csv('/home/jun/CICIDS2019/TrainCIC2019.csv')
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
train = encoding_labels('train', train)
print("DONE!")
print("-------------------------")

# Drop num numberic columns ('Flow ID','Source IP','Destination IP', 'Timestamp', 'SimillarHTTP')
print("Dropping ('Flow ID','Source IP','Destination IP', 'Timestamp', 'SimillarHTTP')")
train = drop_non_numberic_columns(train)
print("DONE!")
print("-------------------------")

#Shuffle data for training 
print("Shuffling data")
train = train.sample(frac=1)
print("DONE!")
print("-------------------------")

# Change data to numpy
print("Change data into numpy")
train = train.to_numpy()
print("DONE!")
print("-------------------------")

# Split dataset to data and label
train_day_X = train[:, :81]
train_day_y = train[:, 81]

# Normalize data to range (0, 1)
scaler = MinMaxScaler()
scaler.fit(train_day_X)
train_normalized = scaler.transform(train_day_X)

# Slit training day data to train (80%), val (0.0002) and test (0.1998)
X_train, X_val_test, y_train, y_val_test = train_test_split(train_day_X, train_day_y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.99, random_state=42) 

# Define Deeplearning parameters
batch_size = 10000
learning_rate = 0.01
num_epochs = 100

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

Train_dataset = MyDataset(X_train.astype(np.float32), y_train.astype(np.float32))
Val_dataset = MyDataset(X_val.astype(np.float32), y_val.astype(np.float32))
Test_dataset = MyDataset(X_test.astype(np.float32), y_test.astype(np.float32))

# Create a data loader for batch training
train_loader = torch.utils.data.DataLoader(Train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(Val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(Test_dataset, batch_size=batch_size, shuffle=False)

# Neural Model
# ANN (inputs_size = 81, 128, 256, 512, 256, 128, 13)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(81, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 512)
        self.l4 = nn.Linear(512, 256)
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.Linear(128, 13)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l6(out)
        return out
    
def get_test_loss_acc(net, criterion, data_loader):
    """A simple function that iterates over `data_loader` to calculate the overall loss"""
    testing_loss = []
    testing_acc = []
    avg_loss = 0
    arg_acc = 0
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

# Define model, put into GPU
model = NeuralNet()
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
writer = SummaryWriter()

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Create csv file to plot data
f = open('/home/jun/CICIDS2019/ANN.csv', 'w')
writer = csv.writer(f)
header = ['10000 batch', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
writer.writerow(header)

# Training loop
train_loss_d, train_acc_d, test_loss_d, test_acc_d = [], [], [], []
running_loss = []
running_acc = []
n_total_step = len(train_loader)
for epoch in range(num_epochs):
    # Total loss on all training sample
    total_loss = []
    # Total accuracy on all training sample
    total_acc = []
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
        if i % 1000 == 0:
            # Calculate average train loss on 10000 batch
            avg_train_loss = np.mean(running_loss)
            running_loss.clear()
            # Calculate average train accuracy on 10000 batch
            avg_train_acc = np.mean(running_acc)
            running_acc.clear()
            # Adding batch loss to total loss
            # Calculate average test loss on all val dataset
            avg_test_loss = get_test_loss_acc(model, criterion, val_loader)['loss']
            # Calculate average train accuracy on all val dataset
            avg_test_acc = get_test_loss_acc(model, criterion, val_loader)['acc']
            total_loss.append(avg_train_loss)
            # Adding batch accuracy to total accuracy
            total_acc.append(avg_train_acc)
            # Adding data to plot after training
            train_loss_d.append(avg_train_loss) 
            train_acc_d.append(avg_train_acc) 
            test_loss_d.append(avg_test_loss) 
            test_acc_d.append(avg_test_acc) 
            print(f'epoch {epoch + 1}/{num_epochs}, step [{i+1}/{n_total_step}], train_loss = {avg_train_loss:.3f}, train_acc = {avg_train_acc:.3f}, test_loss = {avg_test_loss:.3f}, test_acc = {avg_test_acc:.3f}')
            # Write plot data to csv file
            plot_data = [i+1,avg_train_loss,avg_train_acc,avg_test_loss,avg_test_acc]
            writer.writerow(plot_data)
    scheduler.step()
    print('Epoch: [{}/{}], train_Loss: {:.3f}, train_accuracy: {:.3f}'.format(epoch + 1, num_epochs, np.mean(total_loss), np.mean(total_acc)))
    PATH = '/home/jun/CICIDS2019/saved_model/ANN_' + str(epoch+1) + 'epoch.pth'
    torch.save(model.state_dict(), PATH)


