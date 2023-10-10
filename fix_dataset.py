# Import library for load data
import ray
ray.init(num_cpus=24)
import modin.pandas as pd
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import csv
# Import deeplearning library
import time
import os
import glob
import copy
import numpy as np 
import argparse
from CustomFunction import *
from model_define import *

#----------------PARSE PARAMETER-----------------
parser = argparse.ArgumentParser(description='This program will take raw dataset of CICIDS2017, CICIDS2018, CICIDS2019 and turn it into clean all_attack dataset and ddos only dataset')
# parser.add_argument: Add parameter to program
WORK_PATH = os.getcwd()
parser.add_argument('--dataset', help='Raw dataset folder', default= '/home/jun/DDoSDeepLearningProject/Dataset')
parser.add_argument('--fixed', help='Fixed dataset folder', default='/home/jun/DDoSDeepLearningProject/dataset_all_attack') 
parser.add_argument('--ddosonly', help='DDoS only dataset folder', default='/home/jun/DDoSDeepLearningProject/dataset_ddos') 
args = parser.parse_args()
#----------------Define variable based of args----------

raw_dataset_PATH = args.dataset
fixed_dataset_PATH = args.fixed
ddos_dataset_PATH =  args.ddosonly
os.makedirs(fixed_dataset_PATH, exist_ok=True)
os.makedirs(ddos_dataset_PATH, exist_ok=True)
# Make folder for validate data file
print('-------------PARAMETERS-------------')
print('Raw dataset folder PATH:',raw_dataset_PATH)
print('Fixed dataset (all attack) PATH:', fixed_dataset_PATH)
print('DDoS dataset (DDoS attack only) PATH:', ddos_dataset_PATH)

# Print folder and file list
folder = raw_dataset_PATH
csv_file = csv_file_list(folder)
for folder in csv_file.keys():
    print(folder)
    for file in csv_file[folder]:
        print(f'\t{file}')
        
CICIDS2019_columns = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
                'Destination Port', 'Protocol', 'Timestamp', 'Flow Duration',
                'Total Fwd Packets', 'Total Backward Packets',
                'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                'Fwd Packet Length Max', 'Fwd Packet Length Min',
                'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                'Bwd Packet Length Max', 'Bwd Packet Length Min',
                'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
                'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
                'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
                'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
                'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
                'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
                'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
                'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
                'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
                'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
                'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
                'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
                'Idle Std', 'Idle Max', 'Idle Min', 'SimillarHTTP', 'Inbound', 'Label']

CICIDS2018_columns = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
                'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
                'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
                'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
                'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
                'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
                'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
                'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
                'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
                'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
                'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
                'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
                'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
                'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
                'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
                'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
                'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
                'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
                'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
                'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
                'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

CICIDS2018_columns_fixed = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP',
                'Destination Port', 'Protocol', 'Timestamp', 'Flow Duration',
                'Total Fwd Packets', 'Total Backward Packets',
                'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                'Fwd Packet Length Max', 'Fwd Packet Length Min',
                'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                'Bwd Packet Length Max', 'Bwd Packet Length Min',
                'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
                'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
                'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
                'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
                'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
                'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
                'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
                'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
                'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
                'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
                'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
                'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
                'Idle Std', 'Idle Max', 'Idle Min', 'Label']

CICIDS2018_columns_fixed_2 = [
                'Destination Port', 'Protocol', 'Timestamp', 'Flow Duration',
                'Total Fwd Packets', 'Total Backward Packets',
                'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                'Fwd Packet Length Max', 'Fwd Packet Length Min',
                'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                'Bwd Packet Length Max', 'Bwd Packet Length Min',
                'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
                'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
                'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
                'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
                'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
                'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
                'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
                'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
                'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
                'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
                'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
                'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
                'Idle Std', 'Idle Max', 'Idle Min', 'Label']

CICIDS2017_columns = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP',
                'Destination Port', 'Protocol', 'Timestamp', 'Flow Duration',
                'Total Fwd Packets', 'Total Backward Packets',
                'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                'Fwd Packet Length Max', 'Fwd Packet Length Min',
                'Fwd Packet Length Mean', 'Fwd Packet Length Std',
                'Bwd Packet Length Max', 'Bwd Packet Length Min',
                'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
                'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
                'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
                'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
                'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
                'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
                'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
                'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
                'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
                'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
                'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
                'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk',
                'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
                'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
                'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean',
                'Idle Std', 'Idle Max', 'Idle Min', 'Label']

print("CICIDS2019 columns:", len(CICIDS2019_columns))
print("CICIDS2018 columns: {} and {}".format(len(CICIDS2018_columns_fixed), len(CICIDS2018_columns_fixed_2)))
print("CICIDS2017 columns:", len(CICIDS2017_columns))

print('-------------LOADING DATASET-------------')
data = {}
for dataset in csv_file.keys():
    data_dict = {}
    print(f"Start loading folder {dataset}")
    for file in csv_file[dataset]:
        print(f"\tLoading {file}")
        data_key =file.split('/')[-1].split('.')[0] 
        print(data_key)
        data_dict.update({file.split('/')[-1].split('.')[0]:pd.read_csv(file, encoding='cp1252')})
        print("\tDONE!")
    data.update({dataset:data_dict})
print("ALL DONE!")

# Print columns number in each file
for dataset, data_dict in data.items():
    print(dataset)
    for csv_name, csv_data in data_dict.items():
        print(f"\t {csv_name}.csv have {len(csv_data.columns)} columns")
        

# Fix columns name of all dataset
print('-------------FIXING COLUMNS NAME-------------')
for dataset, data_dict in data.items():
    for csv_name, csv_data in data_dict.items():
        csv_data = fix_columns_name(csv_data)
for csv_name, csv_data in data['CICIDS2018'].items():
    try:
        csv_data.columns = CICIDS2018_columns_fixed
    except:
        csv_data.columns = CICIDS2018_columns_fixed_2
print('DONE!')

print('Drop Thursday-WorkingHours-Morning-WebAttacks because of error label')
del data['CICIDS2017']['Thursday-WorkingHours-Morning-WebAttacks']

# Print labels of each file
for dataset, data_dict in data.items():
    print(dataset)
    for csv_name, csv_data in data_dict.items():
        print(f"\t {csv_name}.csv have labels: {set(csv_data['Label'])}")

# Print drop columns in each file
for dataset, data_dict in data.items():
    print(dataset)
    for csv_name, csv_data in data_dict.items():
        this_columns = csv_data.columns
        not_included = []
        for column in this_columns:
            if column not in CICIDS2018_columns_fixed_2:
                not_included.append(column)
        print(f"\t Drop {len(not_included)} columns: {not_included} from {csv_name}.csv")

# Create drop columns list
CICIDS2019_drop_list = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'Fwd Header Length.1', 'SimillarHTTP', 'Inbound']
CICIDS2018_drop_list = ['Destination Port', 'Timestamp']
CICIDS2018_drop_list_2 = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp']
CICIDS2017_drop_list = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'Fwd Header Length.1']


# Drop columns
print('-------------DROPPING COLUMNS TO 78-------------')
for csv_name, csv_data in data['CICIDS2018'].items():
    try:
        csv_data.drop(CICIDS2018_drop_list_2, inplace=True, axis=1)
        print(f'{csv_name} have: {len(csv_data.columns)} columns')
    except:
        csv_data.drop(CICIDS2018_drop_list, inplace=True, axis=1)
        print(f'{csv_name} have: {len(csv_data.columns)} columns')
        print(f"Error {csv_name}")
for csv_name, csv_data in data['CICIDS2017'].items():
    try:
        csv_data.drop(CICIDS2017_drop_list, inplace=True, axis=1)
        print(f'{csv_name} have: {len(csv_data.columns)} columns')
    except:
        print(f"Error {csv_name}")
for csv_name, csv_data in data['CICIDS2019'].items():
    try:
        csv_data.drop(CICIDS2019_drop_list, inplace=True, axis=1)
        print(f'{csv_name} have: {len(csv_data.columns)} columns')
    except:
        print(f"Error {csv_name}")
print('DONE!')
        
# Merge data
print('-------------MERGING DATA-------------')
merged = {}
for dataset, data_dict in data.items():
    merged.update({dataset:pd.concat(data_dict.values(), ignore_index=True, sort=False)})
print('DONE!')

# Change label for CICIDS2018 to fit CICIDS2017
CICIDS2018_labels = {'DoS attacks-SlowHTTPTest':'DoS Slowhttptest', 'DoS attacks-GoldenEye':'DoS GoldenEye', 'DoS attacks-Slowloris':'DoS slowloris','DoS attacks-Hulk':'DoS Hulk', 'Benign': 'BENIGN'}
for label_old, label_new in CICIDS2018_labels.items():
        merged['CICIDS2018']['Label'] = merged['CICIDS2018']['Label'].replace(label_old, label_new)

# Create ddos only dataset
print('-------------CREATING DDOS ONLY DATASET-------------')
ddos_labels = {
    'CICIDS2019': ['Syn', 'DrDoS_NetBIOS', 'DrDoS_SNMP', 'DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_UDP', 'UDP-lag', 'DrDoS_NTP', 'BENIGN', 'WebDDoS', 'DrDoS_MSSQL', 'DrDoS_SSDP', 'TFTP'],
    'CICIDS2018' : ['DDOS attack-LOIC-UDP', 'BENIGN', 'DoS Slowhttptest', 'DoS slowloris', 'DoS GoldenEye', 'DDOS attack-HOIC', 'DoS Hulk', 'DDoS attacks-LOIC-HTTP'],
    'CICIDS2017' : ['BENIGN', 'DoS Slowhttptest', 'DoS slowloris', 'DoS GoldenEye', 'DDoS', 'DoS Hulk']
}
ddos_only = {}
for dataset, data_ddos in merged.items():
    ddos_only.update({dataset:data_ddos.loc[data_ddos['Label'].isin(ddos_labels[dataset])]})
print('DONE!')
merged['CICIDS2018'] = merged['CICIDS2018'][merged['CICIDS2018'].Label != 'Label']
# print out dataset labels
for dataset, data_merged in merged.items():
    print(f"\t {dataset} (all_type_attack) have labels: {set(data_merged['Label'])}")
for dataset, data_merged in ddos_only.items():
    print(f"\t {dataset} (ddos_only) have labels: {set(data_merged['Label'])}")
    

('-------------WRITING TO CSV FILE-------------')
for dataset, data_merged in merged.items():
    merged_PATH =fixed_dataset_PATH + '/' + dataset + '.csv'
    print(f'Writing to {merged_PATH}')
    data_merged.to_csv(merged_PATH,index=False) 
    print('DONE!')
for dataset, data_merged in ddos_only.items():
    print('Writing to {}'.format(dataset))
    merged_PATH =ddos_dataset_PATH + '/' + dataset + '.csv'
    print(f'Writing to {merged_PATH}')
    data_merged.to_csv(merged_PATH,index=False) 
    print('DONE!')
print('ALL DONE!')
