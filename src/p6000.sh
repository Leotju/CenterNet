python main.py ctdet --arch fatnetdasppdcn --dataset pascal --gpus 0,1 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_pascal_384_daspp_dcn_ds4
python test.py ctdet --arch fatnetdasppdcn --dataset pascal --input_res 384 --exp_id fatnet_pascal_384_daspp_dcn_ds4 --resume

python main.py ctdet --arch fatnetdasppdcn --dataset pascal --gpus 0,1 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_pascal_384_daspp_dcn_ds4


python main.py ctdet --arch fatnetdasppdcndla --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lr_10x
python test.py ctdet --arch fatnetdasppdcndla --dataset pascal --down_ratio 4 --input_res 384 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lr_10x --resume


python main.py ctdet --arch fatnetdasppdcndla --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lr_10x


python main.py ctdet --arch fatnetdasppdcndlalk --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lk_lr_10x
python test.py ctdet --arch fatnetdasppdcndlalk --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lk_lr_10x --resume



p40 123
python main.py ctdet --arch fatnetdasppdcndlalk416 --dataset pascal --gpus 2,3 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lk_416_lr_10x
python test.py ctdet --arch fatnetdasppdcndlalk416 --dataset pascal --gpus 2,3 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lk_416_lr_10x --resume


p6000 123
python main.py ctdet --arch fatnetdasppdcndlalkdr --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lk_dr_lr_10x
python test.py ctdet --arch fatnetdasppdcndlalkdr --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lk_dr_lr_10x


python main.py ctdet --arch resdcn_18 --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id resnet_18_dcn_lr_10x
python test.py ctdet --arch resdcn_18 --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id resnet_18_dcn_lr_10x --resume



vpa hello
python main.py ctdet --arch fatnetdasppdcndlalkse --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lk_se_lr_10x
python test.py ctdet --arch fatnetdasppdcndlalkse --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla_lk_se_lr_10x --resume

93
python main.py ctdet --arch fatnetfrndasppdcndlalk416dr --dataset pascal --gpus 0,1,2,3,4,5,6,7 --down_ratio 1 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_384_daspp_dcn_ds4_dla_lk_416_dr_lr_10x
python test.py ctdet --arch fatnetfrndasppdcndlalk416dr --dataset pascal --gpus 0,1,2,3,4,5,6,7 --down_ratio 1 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_384_daspp_dcn_ds4_dla_lk_416_dr_lr_10x --resume

python main.py ctdet --arch fatnetfrndasppdcndlalk416se37 --dataset pascal --gpus 0,1,2,3,4,5,6,7 --down_ratio 1 --input_res 192 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_192_daspp_dcn_ds4_dla_lk_416_se_37_lr_10x
