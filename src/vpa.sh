python main.py ctdet --arch fatnetfrnmbdasppdcn --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_frn_pascal_96_mb_daspp_dcn

python test.py ctdet --arch fatnetfrnmbdasppdcn --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --exp_id fatnet_frn_pascal_96_mb_daspp_dcn

