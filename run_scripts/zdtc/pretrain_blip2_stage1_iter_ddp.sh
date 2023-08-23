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
        --cfg-path lavis/projects/zdtc/pretrain_zdtc_blip2_stage1_iter.yaml
fi