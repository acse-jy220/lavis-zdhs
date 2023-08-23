import io
import json
import os
import ctypes
import multiprocessing
from flask import Flask, request
import argparse
import torch

from server.server_model import *

APP = Flask(__name__)

@APP.route('/health', methods=['GET'])
def health():
    # 检查模型是否预热完成
    if status.get() != 'ok':
        return json.dumps({"health": "false"}), 500
    return json.dumps({"health": "true"}), 200

@APP.route('/infer', methods=['POST'])
def infer_image_func():
    # 检查模型是否预热完成
    if status.get() != 'ok':
        return "service is busy", 500
    # 推理时上锁
    ok = lock.acquire(block=False)
    if not ok:
        return "service is busy", 500

    try:
        ret = {
            'now_context': None,
            'full_context': None,
        }
        picture = request.files.get('picture')
        question = request.form.get('question')
        context = request.form.get('context', '')
        if(question is None or picture is None):
            return  "输入有误", 500
        kwarg = dict(
            use_nucleus_sampling = str2bool(request.form.get('sampling','false')),
            num_beams = int(request.form.get('num_beams','5')),
            top_p = float(request.form.get('top_p','0.9')),
            temperature = float(request.form.get('temperature','1')),
            num_captions = int(request.form.get('num_captions','1')),
            repetition_penalty = float(request.form.get('repetition_penalty','1.5')),
            length_penalty = float(request.form.get('length_penalty','1')),
            max_length = int(request.form.get('max_length','500')),
            max_new_tokens = int(request.form.get('max_new_tokens','250')),
        )
        deterministic = str2bool(request.form.get('deterministic','false'))
        context = truncate_context(context, kwarg['max_length'], kwarg['max_new_tokens'])

        try:
            picture_byte_stream = io.BytesIO(picture.read())
            Image.open(picture_byte_stream).convert('RGB')
        except:
            return "输入图片文件有误", 500
        
        kwarg.update({"picture": picture_byte_stream, "question": question, "context":context, "deterministic":deterministic})

        p2.send(kwarg)

        generated_text = p2.recv()
        if(type(generated_text) == list):
            generated_text = generated_text[0]

        full_context = context + "###问题：\n{}\n\n###答案：{}".format(question, generated_text)
        
        ret['now_context'] = generated_text
        ret['full_context'] = full_context

        return json.dumps(ret, indent=4, ensure_ascii=False)
    finally:
        lock.release()

def str2bool(s):
    if(s == 'true' or s == 'True'):
        return True
    else:
        return False

def worker(args, status, pipe):
    service_object = Server_Model(args, status)
    while True:
        dic = pipe.recv()
        picture = dic.pop('picture')
        question = dic.pop('question')
        deterministic = dic.pop('deterministic')
        context = dic.pop('context')
        pipe.send(service_object.generate(picture, question, context, deterministic, dic))

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn',force=True)
    # 加载模型
    p1, p2 = multiprocessing.Pipe()
    lock = multiprocessing.Lock()
    status = multiprocessing.Manager().Value(ctypes.c_char_p, 'init')

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()

    multiprocessing.Process(target=worker, args=(args, status, p1,)).start()
    APP.run(host="0.0.0.0", port=8787, threaded=True)