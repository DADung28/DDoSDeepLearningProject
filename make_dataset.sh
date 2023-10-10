#!/bin/bash
# python3 (/home/jun/DDoSDeepLearningProject/fix_dataset.py --dataset <raw dataset folder> --fixed <fixed dataset folder> --ddosonly <ddos only dataset folder>)
python3 /home/DDoSDeepLearningProject/fix_dataset.py --dataset /home/DDoSDeepLearningProject/Dataset/ --fixed /home/DDoSDeepLearningProject/dataset_all_attack --ddosonly /home/DDoSDeepLearningProject/dataset_ddos 
