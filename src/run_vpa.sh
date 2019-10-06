#!/usr/bin/env bash
python main.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal07 --num_epochs 70 --lr_step 45,60
python main.py ctdet --exp_id pascal_resdcn101_384 --arch resdcn_101 --dataset pascal07 --num_epochs 70 --lr_step 45,60
python main.py ctdet --exp_id pascal_vgg16_384 --arch vggdcn_16 --dataset pascal07 --num_epochs 70 --lr_step 45,60
