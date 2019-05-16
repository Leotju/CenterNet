python main.py ctdet --arch fatnetdasppdcndla --dataset pascal --gpus 2,3 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla

python main.py ctdet --arch fatnetfrndladasppdcn --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 192 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_frn_pascal_192_daspp_dcn_dla


python test.py ctdet --arch fatnetdasppdcndla --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla --resume