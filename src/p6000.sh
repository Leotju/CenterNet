python main.py ctdet --arch fatnetdasppdcn --dataset pascal --gpus 0,1 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_pascal_384_daspp_dcn_ds4
python test.py ctdet --arch fatnetdasppdcn --dataset pascal --input_res 384 --exp_id fatnet_pascal_384_daspp_dcn_ds4 --resume



python main.py ctdet --arch fatnetdasppdcn --dataset pascal --gpus 0,1 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_pascal_384_daspp_dcn_ds4
