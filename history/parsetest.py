import argparse


parser = argparse.ArgumentParser(description='This program will take input file as CICFLowMeter flow information and output data as analyzed numpy array in to csv file')

# parser.add_argument: Add parameter to program
parser.add_argument('-i', '--input', help='Take CICIDS csv dataset as input file (Input file path)') # Required parameter
parser.add_argument('-o', '--output', help='Output transformed numpy data into csv file (Output file path)') 
args = parser.parse_args() 

print('inout'+args.input)
print('output'+args.output)
