#!/usr/bin/env bash
sleep 4h
python main.py ctdet --exp_id coco_vggdcn18 --arch vgg_16 --gpus 0,1 --num_workers 8 --dataset coco_tiny --batch_size 16 --lr 0.625e-4
