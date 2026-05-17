#!/bin/bash

COUNT=50
for ((i=0;i<$COUNT;i++))
do
    python validate_baseline.py --tag test_baseline --pretrain-weight "/home/dd/DL/DataDistillation/DiM/results/match_kp/model_dict_convnet.pth" --batch-size 50 --ipc 50 --seed ${i}
done