master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_port=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
local_rank=${SLURM_LOCALID}
global_rank=${SLURM_PROCID}
node_rank=${SLURM_NODEID}
node_num=${SLURM_NNODES}
gpu_per_node=${SLURM_NTASKS_PER_NODE}


if [[ $local_rank == 0 ]]; then
    echo master_addr: $master_addr
    echo master_port: $master_port
    echo node_num: $node_num
    echo gpu_per_node: $gpu_per_node
    echo node_rank: $node_rank
    echo local_rank: $local_rank
    echo global_rank $global_rank

    export UNBUFFERED=1

    python -m torch.distributed.run \
        --nnodes=${node_num} \
        --node_rank=${node_rank} \
        --nproc_per_node=${gpu_per_node} \
        --master_addr=${master_addr} \
        --master_port=${master_port} \
        train.py \
        --cfg-path lavis/projects/zdtc/pretrain_zdtc_blip2_instruct.yaml
fi

# python -m torch.distributed.run \
#     --nproc_per_node=8 \
#     --master_port=29005 \
#     train.py \
#     --cfg-path lavis/projects/zdtc/pretrain_zdtc_blip2_instruct.yaml

# nohuo python -m torch.distributed.run \
#     --nproc_per_node=8 \
#     --master_port=29005 \
#     train.py \
#     --cfg-path lavis/projects/zdtc/pretrain_zdtc_blip2_instruct.yaml \
#     > /home/yexiaohan/code/instruct_0713_cn.out 2>&1 &