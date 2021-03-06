python main.py ctdet --arch fatnetdasppdcndla --dataset pascal --gpus 2,3 --down_ratio 4 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla

python main.py ctdet --arch fatnetfrndladasppdcn --dataset pascal --gpus 0,1 --down_ratio 1 --input_res 192 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_frn_pascal_192_daspp_dcn_dla
python test.py ctdet --arch fatnetfrndladasppdcn --dataset pascal --down_ratio 1 --input_res 96 --exp_id fatnet_frn_pascal_192_daspp_dcn_dla --resume
python test.py ctdet --arch fatnetfrndladasppdcn --dataset pascal --down_ratio 1 --input_res 192 --exp_id fatnet_frn_pascal_192_daspp_dcn_dla --resume
python test.py ctdet --arch fatnetfrndladasppdcn --dataset pascal --down_ratio 1 --input_res 384 --exp_id fatnet_frn_pascal_192_daspp_dcn_dla --resume

python main.py ctdet --arch fatnetfrnpre --dataset pascal --gpus 2,3 --down_ratio 1 --input_res 96 --num_epochs 70 --lr_step 45,60 --batch_size 32 --exp_id fatnet_frn_pascal_96_daspp_dcn_branch_pre
python test.py ctdet --arch fatnetfrnpre --dataset pascal --gpus 2,3 --down_ratio 1 --input_res 96 --num_epochs 70 --lr_step 45,60 --batch_size 32 --exp_id fatnet_frn_pascal_96_daspp_dcn_branch_pre --resume


python test.py ctdet --arch fatnetdasppdcndla --dataset pascal --gpus 0,1 --down_ratio 4 --input_res 384 --exp_id fatnet_pascal_384_daspp_dcn_ds4_dla --resume

python main.py ctdet --arch fatnetfrndladasppdcnatt --dataset pascal --gpus 2,3 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-2 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_att_lr100x
python test.py ctdet --arch fatnetfrndladasppdcnatt --dataset pascal --gpus 2,3 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-2 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_att_lr100x --resume

python main.py ctdet --arch fatnetfrndladasppdcnlk --dataset pascal --gpus 2,3 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x
python test.py ctdet --arch fatnetfrndladasppdcnlk --dataset pascal --gpus 2,3 --down_ratio 1 --input_res 96 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_96_daspp_dcn_dla_lk_lr10x --resume



python main.py ctdet --arch fatnetfrndasppdcndlalk416se --dataset pascal --gpus 0,1,2,3 --down_ratio 1 --input_res 192 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_192_daspp_dcn_dla_lk_se_416_lr10x
python test.py ctdet --arch fatnetfrndasppdcndlalk416se --dataset pascal --gpus 0,1,2,3 --down_ratio 1 --input_res 192 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_192_daspp_dcn_dla_lk_se_416_lr10x --resume


python main.py ctdet --arch fatnetfrnmpdcn --dataset pascal --gpus 0,1,2,3 --down_ratio 1 --input_res 192 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_192_multi_pool_dcn_lr10x
python test.py ctdet --arch fatnetfrnmpdcn --dataset pascal --gpus 0,1,2,3 --down_ratio 1 --input_res 192 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_pascal_192_multi_pool_dcn_lr10x --resume


python main.py ctdet --arch fatnetfrnblockmpdcn --wh_weight 0.02 --dataset pascal --gpus 0,1,2,3 --down_ratio 1 --input_res 192 --num_epochs 210 --lr_step 135,180 --batch_size 32 --lr 1.25e-3 --exp_id fatnet_frn_block_pascal_192_multi_pool_dcn_lr10x_wh002


python main.py ctdet --arch resdcn_18 --dataset pascal --gpus 0,1,2,3 --down_ratio 1 --input_res 384 --num_epochs 70 --lr_step 45,60 --batch_size 128 --wh_weight 0.02 --lr 5e-4 --exp_id resnet_18_dcn_wh_002_d1_dcn_up



