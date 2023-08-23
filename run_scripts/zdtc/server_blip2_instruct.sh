# CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.run --master_port 29518 --nproc_per_node=1 server_flask_stream.py --cfg-path lavis/projects/zdtc/eval_zdtc_blip2_stage2.yaml

python server_flask_stream.py --cfg-path lavis/projects/zdtc/server_zdtc_blip2_instruct.yaml
python server_flask.py --cfg-path lavis/projects/zdtc/server_zdtc_blip2_instruct.yaml
