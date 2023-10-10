#/bin/bash

# Trained 2017
python3 validation.py --device 0 --epoch 50 --dataset ./dataset_ddos/CICIDS2017.csv --model CNN --ver 0 --trained_type CICIDS2017_anomaly --test_type CICIDS2017_anomaly
python3 validation.py --device 0 --epoch 50 --dataset ./dataset_ddos/CICIDS2018.csv --model CNN --ver 0 --trained_type CICIDS2017_anomaly --test_type CICIDS2018_anomaly
python3 validation.py --device 0 --epoch 50 --dataset ./dataset_ddos/CICIDS2019.csv --model CNN --ver 0 --trained_type CICIDS2017_anomaly --test_type CICIDS2019_anomaly
# Trained 2018
python3 validation.py --device 0 --epoch 50 --dataset ./dataset_ddos/CICIDS2018.csv --model CNN --ver 0 --trained_type CICIDS2018_anomaly --test_type CICIDS2018_anomaly
python3 validation.py --device 0 --epoch 50 --dataset ./dataset_ddos/CICIDS2017.csv --model CNN --ver 0 --trained_type CICIDS2018_anomaly --test_type CICIDS2017_anomaly
python3 validation.py --device 0 --epoch 50 --dataset ./dataset_ddos/CICIDS2019.csv --model CNN --ver 0 --trained_type CICIDS2018_anomaly --test_type CICIDS2019_anomaly
# Trained 2019
python3 validation.py --device 0 --epoch 25 --dataset ./dataset_ddos/CICIDS2019.csv --model CNN --ver 0 --trained_type CICIDS2019_anomaly --test_type CICIDS2019_anomaly
python3 validation.py --device 0 --epoch 25 --dataset ./dataset_ddos/CICIDS2017.csv --model CNN --ver 0 --trained_type CICIDS2019_anomaly --test_type CICIDS2017_anomaly
python3 validation.py --device 0 --epoch 25 --dataset ./dataset_ddos/CICIDS2018.csv --model CNN --ver 0 --trained_type CICIDS2019_anomaly --test_type CICIDS2018_anomaly
# Trained Combinated dataset
python3 validation.py --device 0 --epoch 1 --dataset ./dataset_ddos/CICIDS2019.csv --model CNN --ver 0 --trained_type combinated --test_type CICIDS2019_anomaly
python3 validation.py --device 0 --epoch 1 --dataset ./dataset_ddos/CICIDS2017.csv --model CNN --ver 0 --trained_type combinated --test_type CICIDS2017_anomaly
python3 validation.py --device 0 --epoch 1 --dataset ./dataset_ddos/CICIDS2018.csv --model CNN --ver 0 --trained_type combinated --test_type CICIDS2018_anomaly
python3 validation_combinated.py --device 0 --epoch 1  --model CNN --ver 0 --trained_type combinated --test_type CICIDS2018_anomaly