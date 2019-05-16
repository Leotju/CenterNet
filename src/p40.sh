python main.py ctdet --arch fatnetdasppdcndla --dataset pascal --gpus 2,3 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla


python test.py ctdet --arch fatnetdasppdcndla --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla --resume