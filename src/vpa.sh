python main.py ctdet --arch fatnetfrnmbdasppdcn --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_frn_pascal_96_mb_daspp_dcn
python test.py ctdet --arch fatnetfrnmbdasppdcn --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --exp_id fatnet_frn_pascal_96_mb_daspp_dcn



python main.py ctdet --arch fatnetfrndladasppdcnatt --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_att_lr10x
python test.py ctdet --arch fatnetfrndladasppdcnatt --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_att_lr10x --resume


python main.py ctdet --arch fatnetfrndasppdcndlalk416cbam --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 16 --lr 1.25e-3 --exp_id fatnetfrndasppdcndlalk416cbam_lr10x_wh002 --wh_weight 0.02
python test.py ctdet --arch fatnetfrndasppdcndlalk416cbam --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 16 --lr 1.25e-3 --exp_id fatnetfrndasppdcndlalk416cbam_lr10x_wh002 --wh_weight 0.02 --resume


python main.py ctdet --arch fatnetfrndasppdcndlalk416cbam --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 16 --lr 1.25e-3 --exp_id fatnetfrndasppdcndlalk416cbam_sa21_ca2_lr10x_wh002 --wh_weight 0.02
python test.py ctdet --arch fatnetfrndasppdcndlalk416cbam --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 16 --lr 1.25e-3 --exp_id fatnetfrndasppdcndlalk416cbam_sa21_ca2_lr10x_wh002 --wh_weight 0.02 --resume



python main.py ctdet --exp_id resdcnbu --arch resdcnbu_18 --down_ratio 4 --dataset pascal --input_res 384 --num_epochs 70 --lr_step 45,60  --gpus 0,1
python test.py ctdet --exp_id resdcnbu --arch resdcnbu_18 --down_ratio 4 --dataset pascal --input_res 384 --num_epochs 70 --lr_step 45,60  --gpus 0,1 --resume

python main.py ctdet --exp_id resdcnbu18_512 --arch resdcnbu_18 --down_ratio 4 --dataset pascal --input_res 512 --num_epochs 70 --lr_step 45,60  --gpus 2
python test.py ctdet --exp_id resdcnbu18_512 --arch resdcnbu_18 --down_ratio 4 --dataset pascal --input_res 512 --num_epochs 70 --lr_step 45,60  --gpus 0,1,2,3 --resume



python main.py ctdet --arch fatnetfrnpooldladcn --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 16 --lr 1.25e-3 --exp_id fatnet_frn_pool_dla_dcn_lr10x_wh002_96_no_dil --wh_weight 0.02

python main.py ctdet --arch fatnetfrnpooldladcn --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 16 --lr 1.25e-3 --exp_id fatnet_frn_pool_dla_dcn_lr10x_wh002_96_no_dil_ch2x --wh_weight 0.02

python main.py ctdet --arch resdcn_18 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 384 --num_epochs 70 --lr_step 45,60 --batch_size 16 --wh_weight 0.02 --lr 0.625e-4 --exp_id resnet_18_dcn_wh_002_d1

python main.py ctdet --arch resdcn_18 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 384 --num_epochs 70 --lr_step 45,60 --batch_size 16 --wh_weight 0.02 --lr 0.625e-4 --exp_id resnet_18_dcn_wh_002_d1_dcn_up


python main.py ctdet --arch vgg_16 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 384 --num_epochs 70 --lr_step 45,60 --batch_size 16 --wh_weight 0.02 --lr 0.625e-4 --exp_id vgg_dcn_wh_002_d1_up_bil

python main.py ctdet --arch vgg_16 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 384 --num_epochs 70 --lr_step 45,60 --batch_size 16 --wh_weight 0.1 --lr 0.625e-4 --exp_id vgg_dcn_wh_01_d1_up_bil

python main.py ctdet --arch fatnet_frn_tl_conv --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 384 --num_epochs 70 --lr_step 45,60 --batch_size 16 --wh_weight 0.02 --lr 0.625e-4 --exp_id fatnet_frn_tl_conv_wh002

python main.py ctdet --arch vgg_16 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 384 --num_epochs 70 --lr_step 45,60 --batch_size 16 --wh_weight 0.005 --lr 0.625e-4 --exp_id vgg_dcn_wh_0005_d1_up_bil

#python main.py ctdet --arch vggdla_16 --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 384 --num_epochs 70 --lr_step 45,60 --batch_size 2 --wh_weight 0.02 --lr 0.625e-4 --exp_id vgg_no_pool_wh_002_d1

python main.py ctdet --arch fatnet_frn_tl_conv --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 384 --num_epochs 70 --lr_step 45,60 --batch_size 16 --wh_weight 0.02 --lr 0.625e-4 --exp_id fatnet_frn_tl_conv_wh002

