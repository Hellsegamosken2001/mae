python -m torch.distributed.launch --nproc_per_node=2 main_finetune.py \
    --batch_size 512 \
    --input_size 32 \
    --nb_classes 10\
    --model deit_tiny_patch4 \
    --finetune ../output/train/mae/checkpoint-199.pth  \
    --epochs 200 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ../cifar10/\
    --output_dir ../output/finetune/mae;\