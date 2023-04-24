python  -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 29531 main_pretrain.py \
    --batch_size 512 \
    --input_size 32 \
    --model mae_deit_tiny_patch42\
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 5 \
    --blr 7.5e-4 --weight_decay 0.05 \
    --data_path ../cifar10/\
    --model_ema\
    --output_dir ../output/train/Bmae