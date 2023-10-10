#!/bin/bash

python3 train.py --device 6 --dataset ./dataset_ddos/CICIDS2017.csv --model CNN --ver 0 --epoch_start 0 --scheduler --lr 0.01 --bs 1000 --type CICIDS2017_anomaly
python3 train.py --device 4 --dataset ./dataset_ddos/CICIDS2018.csv --model CNN --ver 0 --epoch_start 0 --scheduler --lr 0.01 --bs 1000 --type CICIDS2018_anomaly
python3 train.py --device 3 --dataset ./dataset_ddos/CICIDS2019.csv --model CNN --ver 0 --epoch_start 0 --scheduler --lr 0.01 --bs 1000 --type CICIDS2019_anomaly
python3 train_combinated.py --device 3 --model CNN --ver 0 --epoch_start 0 --scheduler --lr 0.01 --bs 1000 --type combinated