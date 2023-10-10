# Import library for load data
import numpy as np
import os
import glob
import torch
# The columns name of dataset is contain space, this function will strip it out (data = fix_columns_name(data))
def fix_columns_name(data):
    data.columns = [column_name.strip() for column_name in data.columns]
    return data

# Drop all string data in dataset (data = drop_non_numberic_columns(data))
def drop_non_numberic_columns(data):
    data.drop(['Unnamed: 0', 'Flow ID','Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'SimillarHTTP'], inplace=True, axis=1)
    return data

#Create a copy of pandas frame with dropped NaN and Inf value (data = drop_NaN(data))
def drop_NaN(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    return data

# Return a dictionary of available labels and it number from dataset 'Label' column (labels_dict = available_lable(data))
# Input: dataset
# Output: 
# Traning dataset: {'Syn': 0, 'DrDoS_SNMP': 1, 'DrDoS_UDP': 2, 'DrDoS_DNS': 3, 'TFTP': 4, 'DrDoS_LDAP': 5, 'UDP-lag': 6, 'DrDoS_NTP': 7, 'DrDoS_MSSQL': 8, 'DrDoS_NetBIOS': 9, 'BENIGN': 10, 'WebDDoS': 11, 'DrDoS_SSDP': 12}
# Testing dataset {'Syn': 0, 'NetBIOS': 1, 'UDPLag': 2, 'LDAP': 3, 'MSSQL': 4, 'BENIGN': 5, 'Portmap': 6, 'UDP': 7} 
def available_label(data):
    data_labels = set(data['Label'])
    labels_dict = {}
    for i, label in enumerate(data_labels):
        labels_dict.update({label:i})
    return labels_dict

# Caculate flow labels number and return dictionary of {labels : count}
# train_flow_labels_count = flow_labels_count(data)
def flows_labels_count(data):
    labels = {}
    for flow_label in data['Label']:
        if flow_label in labels.keys():
            labels[flow_label] += 1
        else:
            labels[flow_label] = 0
    labels = dict(sorted(labels.items(), key=lambda x:x[1], reverse=True))
    return labels

# Encoding labels to assigned number for learning
# data = encoding_lables('train', data)
def encoding_labels(dataset_type, data):
    if dataset_type == 'CICIDS2019':
        labels_num = {'BENIGN': 0, 'Syn': 1, 'DrDoS_SNMP': 2, 'DrDoS_UDP': 3, 'DrDoS_DNS': 4, 'TFTP': 5, 'DrDoS_LDAP': 6, 'UDP-lag': 7, 'DrDoS_NTP': 8, 'DrDoS_MSSQL': 9, 'DrDoS_NetBIOS': 10, 'WebDDoS': 11, 'DrDoS_SSDP': 12}
    elif dataset_type == 'CICIDS2019_anomaly':
        labels_num = {'BENIGN': 0, 'Syn': 1, 'DrDoS_SNMP': 1, 'DrDoS_UDP': 1, 'DrDoS_DNS': 1, 'TFTP': 1, 'DrDoS_LDAP': 1, 'UDP-lag': 1, 'DrDoS_NTP': 1, 'DrDoS_MSSQL': 1, 'DrDoS_NetBIOS': 1, 'WebDDoS': 1, 'DrDoS_SSDP': 1}
    elif dataset_type == 'CICIDS2019_test':
        labels_num = {'BENIGN': 0, 'Syn': 1, 'NetBIOS': 10, 'UDPLag': 7, 'LDAP': 6, 'MSSQL': 9, 'Portmap': 13, 'UDP': 3} 
    elif dataset_type == 'CICIDS2018_ddos':
        labels_num = {'BENIGN':0, 'DDOS attack-LOIC-UDP':1, 'DoS Slowhttptest':2, 'DoS slowloris':3, 'DoS GoldenEye':4, 'DDOS attack-HOIC':5, 'DoS Hulk':6, 'DDoS attacks-LOIC-HTTP':7}
    elif dataset_type == 'CICIDS2018_all':
        labels_num = {'BENIGN':0, 'DDOS attack-LOIC-UDP':1, 'DoS Slowhttptest':2, 'DoS slowloris':3, 'DoS GoldenEye':4, 'DDOS attack-HOIC':5, 'DoS Hulk':6, 'DDoS attacks-LOIC-HTTP':7, 'SQL Injection':8, 'Bot':9, 'Brute Force -XSS':10, 'Infilteration':11, 'Brute Force -Web':12, 'SSH-Bruteforce':13, 'FTP-BruteForce':14}
    elif dataset_type == 'CICIDS2018_anomaly':
        labels_num = {'BENIGN':0, 'DDOS attack-LOIC-UDP':1, 'DoS Slowhttptest':1, 'DoS slowloris':1, 'DoS GoldenEye':1, 'DDOS attack-HOIC':1, 'DoS Hulk':1, 'DDoS attacks-LOIC-HTTP':1}
    elif dataset_type == 'CICIDS2017':
        labels_num = {'BENIGN':0, 'DoS Slowhttptest':1, 'DoS slowloris':2, 'DoS GoldenEye':3, 'DDoS':4, 'DoS Hulk':5, 'Heartbleed':6}
    elif dataset_type == 'CICIDS2017_anomaly':
        labels_num = {'BENIGN':0, 'DoS Slowhttptest':1, 'DoS slowloris':1, 'DoS GoldenEye':1, 'DDoS':1, 'DoS Hulk':1, 'Heartbleed':1}  
    for label, number in labels_num.items():
        data['Label'] = data['Label'].replace(label, number)
    return data

# Ploting data
def data_plot(data):
    data_flow_labels_count = flows_labels_count(data)
    for flow_label, flow_number in data_flow_labels_count.items():
        print('Number of {} flow is: {}/{} ({:.2f})'.format(flow_label, flow_number, sum(data_flow_labels_count.values()), flow_number/sum(data_flow_labels_count.values())*100))
    #Create Flow Number based on label Figure
    data_flows_figure_num = plt.figure(figsize = (20, 5))
    # creating the bar plot
    plt.bar(data_flow_labels_count.keys(), data_flow_labels_count.values(), color ='red',
            width = 0.5)
    plt.xlabel("Flows Label")
    plt.ylabel("Number of FLows")
    plt.title("Flows label and number")
    plt.show()

#This function take input as a folder and return all csv file and folder in it as dictionary {<subfolder>:<file>}
def csv_file_list(folder):
    # Get all sub directories in dataset
    dir_list = [dir[0] for dir in os.walk(folder)]
    # Create dictionary for {<folder:file>}
    file = {}
    for dir in dir_list:
        dir_name = dir.split('/')[-1]
        csv_file = glob.glob(dir+'/*.csv') 
        if csv_file != []:
            file.update({dir_name:csv_file})
    return file
# This function will take input as tensor object of label and predicted and caculate confusion matrix of num_classes classification
def ConfusionMatrix(actual,predicted,num_classes):
    c = torch.zeros((num_classes,num_classes))
    for i in range(actual.shape[0]):
        c[actual[i]][predicted[i]] += 1
    return c
# This function will take input of confusion matrix and return actual labels number, accuracy, precision, recall and f1_score
def Validate(c_matrix):
    output_size = c_matrix.shape[0]
    precision = torch.zeros((output_size))
    recall = torch.zeros((output_size))
    f1_score = torch.zeros((output_size))
    actual = torch.zeros((output_size))
    accuracy = 0
    for i in range(output_size):
        actual[i] = torch.sum(c_matrix[i])
        accuracy += c_matrix[i][i]
        recall[i] = c_matrix[i][i]/torch.sum(c_matrix[i])
        precision[i] = c_matrix[i][i]/torch.sum(c_matrix[:,i])
        f1_score[i] = 2*(precision[i]*recall[i])/(precision[i]+recall[i])
    accuracy /= torch.sum(c_matrix)
    return  {'actual': actual, 'accuracy': accuracy, 'precision':precision,'recall':recall,'f1_score':f1_score}
    