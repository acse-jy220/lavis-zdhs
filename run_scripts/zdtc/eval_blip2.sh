CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/zdtc/eval_zdtc_blip2_stage1_retrieval.yaml
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/zdtc/eval_zdtc_blip2_stage1_caption.yaml
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/zdtc/eval_zdtc_blip2_stage2_caption.yaml
CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path lavis/projects/zdtc/eval_zdtc_blip2_instruct.yaml

# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.run \
#     --nproc_per_node=2 \
#     --master_port=29005 \
#     train.py \
#     --cfg-path lavis/projects/zdtc/pretrain_zdtc_blip2_stage1_iter.yaml

# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.run \
#     --nproc_per_node=2 \
#     --master_port=29005 \
#     train.py \
#     --cfg-path lavis/projects/zdtc/pretrain_zdtc_blip2_stage2_iter.yaml

# CUDA_VISIBLE_DEVICES=2,3 \
# python -m torch.distributed.run \
#     --nproc_per_node=2 \
#     --master_port=29005 \
#     train.py \
#     --cfg-path lavis/projects/zdtc/pretrain_zdtc_blip2_instruct.yaml
