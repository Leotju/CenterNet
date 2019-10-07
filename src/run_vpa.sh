#!/usr/bin/env bash

python main.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --batch_size 64 --master_batch 18 --lr 2.5e-4 --gpus 0,1 --num_workers 8 --dataset coco_tiny
