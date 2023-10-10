# Import deeplearning library
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms 
from torch.utils.data import Dataset

# Class to make custom data set from numpy array
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
# Artificial Neural Network with no dropout
class ANN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ANN, self).__init__()
        self.l1 = nn.Linear(input_size, 1000)
        self.l2 = nn.Linear(1000, 2000)
        self.l3 = nn.Linear(2000, 4000)
        self.l4 = nn.Linear(4000, 6000)
        self.l5 = nn.Linear(6000, 8000)
        self.l6 = nn.Linear(8000, 16000)
        self.l7 = nn.Linear(16000, 5000)
        self.l8 = nn.Linear(5000,output_size)
        self.relu = nn.ReLU()
    
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.l6(out)
        out = self.relu(out)
        out = self.l7(out)
        out = self.relu(out)
        out = self.l8(out)
        return out

# Artificial Neural Network with dropout    
class ANND(nn.Module):
    def __init__(self, input_size, output_size):
        super(ANND, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 512)
        self.l4 = nn.Linear(512, 1024)
        self.l5 = nn.Linear(1024,output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l5(out)
        return out

# Deep Neural Network with no dropout
class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 64)
        self.l3 = nn.Linear(64, 128)
        self.l4 = nn.Linear(128, 128)
        self.l5 = nn.Linear(128, 128)
        self.l6 = nn.Linear(128, 64)
        self.l7 = nn.Linear(64, 32)
        self.l8 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.l6(out)
        out = self.relu(out)
        out = self.l7(out)
        out = self.relu(out)
        out = self.l8(out)
        return out

# Deep Neural Network with dropout
class DNND(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNND, self).__init__()
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 512)
        self.l4 = nn.Linear(512, 1028)
        self.l5 = nn.Linear(1028, 512)
        self.l6 = nn.Linear(512, 256)
        self.l7 = nn.Linear(256, 128)
        self.l8 = nn.Linear(128, 64)
        self.l9 = nn.Linear(64, 32)
        self.l10 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l5(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l6(out)
        out = self.relu(out)
        out = self.l7(out)
        out = self.relu(out)
        out = self.l8(out)
        out = self.relu(out)
        out = self.l9(out)
        out = self.relu(out)
        out = self.l10(out)
        return out
    
class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        # call the parent constructor
        super(CNN, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.linear1 = nn.Linear(input_size, 3*40*40)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=40, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=200,kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = nn.Flatten()
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=9800, out_features=5000)
        self.relu3 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=5000, out_features=output_size)
        #self.logSoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.linear1(x)
        x = torch.reshape(x,(-1,3,40,40))
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        # return the output predictions
        return x  

class CNN2(nn.Module):
    def __init__(self, input_size, output_size):
        # call the parent constructor
        super(CNN2, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.linear1 = nn.Linear(input_size, 3*100*100)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=40, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=80,kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv3 = nn.Conv2d(in_channels=80, out_channels=200,kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize flatten layer for FC Layer
        self.flatten = nn.Flatten()
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=16200, out_features=5000)
        self.relu4 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=5000, out_features=output_size)
        #self.logSoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.linear1(x)
        # Reshape x to feed conv layer with 3 channel
        x = torch.reshape(x,(-1,3,100,100))
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        # return the output predictions
        return x  
    
class CNN3(nn.Module):
    def __init__(self, input_size, output_size):
        # call the parent constructor
        super(CNN3, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.linear1 = nn.Linear(input_size, 3*244*244)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=40, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=40, out_channels=80,kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv3 = nn.Conv2d(in_channels=80, out_channels=160,kernel_size=(3, 3))
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv4 = nn.Conv2d(in_channels=160, out_channels=320,kernel_size=(3, 3))
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize flatten layer for FC Layer
        self.flatten = nn.Flatten()
        # initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=54080, out_features=5000)
        self.relu5 = nn.ReLU()
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=5000, out_features=output_size)
        #self.logSoftmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.linear1(x)
        # Reshape x to feed conv layer with 3 channel
        x = torch.reshape(x,(-1,3,244,244))
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x = self.fc2(x)
        # return the output predictions
        return x  

