# Import library for load data
import ray
ray.init(num_cpus=7)
import modin.pandas as pd
import numpy as np 
from CustomFunction import *
import argparse

parser = argparse.ArgumentParser(description='This program will take input file as CICFLowMeter flow information (csv file) and output data as analyzed numpy array for training (csv file)')

# parser.add_argument: Add parameter to program
parser.add_argument('-i', '--input', help='Input file path') # Required parameter
parser.add_argument('-o', '--output', help='Output file path') 
parser.add_argument('-t', '--type', help='Type of dataset [train] or [test]') 

args = parser.parse_arg()

print('-------------ANALYZE DATASET-------------')
#Load dataset
print("Loading dataset, please wait .....")
data = pd.read_csv(args.input)
print("DONE!")
print("-------------------------")

# Fix columns name
print("Start fixing columns name")
data = fix_columns_name(data)
print("DONE!")
print("-------------------------")

# Drop NaN and inf
print("Start dropping NaN and Infinity contained row")
data = drop_NaN(data)
print("DONE!")
print("-------------------------")

# Encoding data dataset label
print("Start encoding flow labels to number")
data = encoding_labels(args.type, data)
print("DONE!")
print("-------------------------")

# Drop num numberic columns ('Flow ID','Source IP','Destination IP', 'Timestamp', 'SimillarHTTP')
print("Dropping ('Flow ID','Source IP','Destination IP', 'Timestamp', 'SimillarHTTP')")
data = drop_non_numberic_columns(data)
print("DONE!")
print("-------------------------")

# Change data to numpy
print("Change data into numpy")
data = data.to_numpy()
data = np.asfarray(data)
print("DONE!")
print("-------------------------")

print("Write to csv file")
np.savetxt(args.output, data, delimiter=',')
print("DONE!")
print("-------------------------")
