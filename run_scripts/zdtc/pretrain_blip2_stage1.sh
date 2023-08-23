python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port 29509 \
    train.py \
    --cfg-path lavis/projects/zdtc/pretrain_zdtc_blip2_stage1.yaml