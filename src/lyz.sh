python main.py ctdet --arch fatnetfrndladasppdcnlk --wh_weight 0.02 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x_wh_002
python test.py ctdet --arch fatnetfrndladasppdcnlk --wh_weight 0.02 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x_wh_002 --resume

python main.py ctdet --arch fatnetfrndladasppdcnlk --wh_weight 0.2 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x_wh_02
python test.py ctdet --arch fatnetfrndladasppdcnlk --wh_weight 0.2 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x_wh_02 --resume



python main.py ctdet --arch vgg_16 --exp_id coco_vgg_384 --batch_size 128 --lr 5e-4 --gpus 0,1,2,3 --num_workers 16 --val_intervals 10 --input_res 384
python test.py ctdet --arch vgg_16 --dataset coco2014 --exp_id coco_vgg_384 --input_res 384 --resume
python demo.py ctdet --arch vgg_16 --dataset coco2014 --exp_id coco_vgg_384 --input_res 384 --resume --demo 'webcam'

python main.py ctdet --arch vggdla_16 --exp_id pascal_vggdla_384 --dataset pascal --gpus 0,1 --val_intervals 10 --input_res 384 --num_epochs 70 --lr_step 45,60
python test.py ctdet --arch vggdla_16 --exp_id pascal_vggdla_384 --dataset pascal --gpus 0,1 --val_intervals 10 --input_res 384 --num_epochs 70 --lr_step 45,60 --resume

python main.py ctdet --exp_id pascal_resdcn18_384 --arch resdcn_18 --dataset pascal --num_epochs 70 --lr_step 45,60



python main.py ctdet --arch resdcn_18 --exp_id coco_res18_384 --batch_size 128 --lr 5e-4 --gpus 0,1 --num_workers 16 --val_intervals 10 --input_res 384
python test.py ctdet --arch resdcn_18 --exp_id coco_res18_384 --batch_size 128 --lr 5e-4 --gpus 0,1 --num_workers 16 --val_intervals 10 --input_res 384 --resume
python test.py ctdet --arch resdcn_18 --dataset coco2014 --exp_id coco_res18_384 --lr 5e-4 --gpus 0 --num_workers 16 --val_intervals 10 --input_res 384 --resume
python test.py ctdet --arch resdcn_18 --dataset coco2014 --exp_id coco_vgg_384 --load_model  /home/leo/Downloads/ctdet_coco_resdcn18.pth

python test.py ctdet --arch resdcn_18 --dataset coco2014 --exp_id coco_res18_384 --input_res 384 --resume



python demo.py ctdet --arch fatnetfrndladasppdcn --dataset pascal --down_ratio 1 --input_res 192 --exp_id fatnet_frn_pascal_192_daspp_dcn_dla --resume --demo 'webcam'


python main.py ctdet --arch fatnetfrndcnpre --wh_weight 0.02 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 70 --lr_step 45,60 --exp_id fatnet_frn_pascal_96_daspp_dcn_pre_wh_002


python main.py ctdet --arch fatnetfrndcnpre --wh_weight 0.02 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 70 --lr_step 45,60 --exp_id fatnet_frn_pascal_96_daspp_dcn_pre_wh_002_lk

python main.py ctdet --arch fatnetfrndcnpre --wh_weight 0.02 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 70 --lr_step 45,60 --exp_id fatnet_frn_pascal_96_daspp_dcn_pre_wh_002_lk_no_res


