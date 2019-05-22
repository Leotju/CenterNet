python main.py ctdet --arch fatnetfrndladasppdcnlk --wh_weight 0.02 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x_wh_002
python test.py ctdet --arch fatnetfrndladasppdcnlk --wh_weight 0.02 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x_wh_002 --resume

python main.py ctdet --arch fatnetfrndladasppdcnlk --wh_weight 0.2 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x_wh_02
python test.py ctdet --arch fatnetfrndladasppdcnlk --wh_weight 0.2 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x_wh_02 --resume



python main.py ctdet --arch vgg_16 --exp_id coco_vgg_384 --batch_size 128 --lr 5e-4 --gpus 0,1,2,3 --num_workers 16 --val_intervals 10 --input_res 384

python main.py ctdet --arch resdcn_18 --exp_id coco_res18_384 --batch_size 128 --lr 5e-4 --gpus 0,1 --num_workers 16 --val_intervals 10 --input_res 384
python test.py ctdet --arch resdcn_18 --exp_id coco_res18_384 --batch_size 128 --lr 5e-4 --gpus 0,1 --num_workers 16 --val_intervals 10 --input_res 384 --resume

