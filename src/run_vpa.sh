#!/usr/bin/env bash
python main.py ctdet --exp_id pascal07_resdcn18_384 --arch resdcn_18 --dataset pascal07 --num_epochs 70 --lr_step 45,60 --gpus 0,1
python main.py ctdet --exp_id pascal07_resdcn18_512 --arch resdcn_18 --dataset pascal07 --input_res 512 --num_epochs 70 --lr_step 45,60 --gpus 0,1
python main.py ctdet --exp_id pascal07_resdcn101_384 --arch resdcn_101 --dataset pascal07 --num_epochs 70 --lr_step 45,60 --gpus 0,1
python main.py ctdet --exp_id pascal07_resdcn101_512 --arch resdcn_101 --dataset pascal07 --input_res 512 --num_epochs 70 --lr_step 45,60 --gpus 0,1
python main.py ctdet --exp_id pascal07_vgg16_384 --arch vggdcn_16 --dataset pascal07 --num_epochs 70 --lr_step 45,60 --gpus 0,1
python main.py ctdet --exp_id pascal07_vgg16_512 --arch vggdcn_16 --dataset pascal07 --input_res 512 --num_epochs 70 --lr_step 45,60 --gpus 0,1

python test.py ctdet --exp_id pascal07_resdcn18_384 --arch resdcn_18 --dataset pascal07 --num_epochs 70 --lr_step 45,60 --resume
python test.py ctdet --exp_id pascal07_resdcn18_512 --arch resdcn_18 --dataset pascal07 --input_res 512 --num_epochs 70 --lr_step 45,60 --resume
python test.py ctdet --exp_id pascal07_resdcn101_384 --arch resdcn_101 --dataset pascal07 --num_epochs 70 --lr_step 45,60 --resume
python test.py ctdet --exp_id pascal07_resdcn101_512 --arch resdcn_101 --dataset pascal07 --input_res 512 --num_epochs 70 --lr_step 45,60 --resume
python test.py ctdet --exp_id pascal07_vgg16_384 --arch vggdcn_16 --dataset pascal07 --num_epochs 70 --lr_step 45,60 --resume
python test.py ctdet --exp_id pascal07_vgg16_512 --arch vggdcn_16 --dataset pascal07 --input_res 512 --num_epochs 70 --lr_step 45,60 --resume
