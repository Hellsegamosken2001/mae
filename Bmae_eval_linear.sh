python -m torch.distributed.launch --nproc_per_node=2 --master_port 29531 main_linprobe.py \
    --batch_size 512 \
    --model deit_tiny_patch4 \
    --nb_classes 10\
    --finetune ../output/train/Bmae/checkpoint-199.pth \
    --epochs 100 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ../cifar10/\
    --output_dir ../output/linear/Bmae