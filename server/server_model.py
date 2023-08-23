import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from lavis.datasets.data_utils import prepare_sample
from lavis.common.logger import MetricLogger

from lavis.common.registry import registry
from PIL import Image
import os

from torch.utils.data.dataloader import default_collate

def parse_args():
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

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def truncate_context(context, max_len, max_new_len):
    context_max_len = max_len - 50 - max_new_len
    context_len = len(context)
    if(context_len <= context_max_len):
        return context
    else:
        return context[context_len-context_max_len : ]

class Server_Model():
    def __init__(self, args, status=None):
        cfg = Config(args)
        setup_logger()

        llm = os.getenv('LLM_MODEL_PATH',None)
        ckpt = os.getenv('CKPT_MODEL_PATH',None)

        if(llm is not None):
            cfg.model_cfg.llm_model = llm
        if(ckpt is not None):
            cfg.model_cfg.finetuned = ckpt

        cfg.pretty_print()

        task = tasks.setup_task(cfg)

        datasets_config = cfg.datasets_cfg

        self.vis_processors = None
        self.aud_processors = None

        for name in datasets_config:
            if(name == 'zdtc_blip2_instruct'):
                dataset_config = datasets_config[name]

                builder = registry.get_builder_class(name)(dataset_config)
                builder.build_processors()

                self.vis_processors = builder.vis_processors["eval"]
                self.aud_processors = builder.text_processors["eval"]

        assert self.vis_processors is not None
        assert self.aud_processors is not None

        self.model = task.build_model(cfg).to('cuda:0')

        status.set('ok')
        print('model is ready')

    def generate(self, image, question, context, deterministic, kwargs):
        if image is not None:
            try:
                image = Image.open(image).convert('RGB')
            except:
                return "输入图片文件有误"
            
            image = self.vis_processors(image)

        question = [question]
        context = [context]

        sample = {}
        sample['image'] = torch.stack([image], dim=0)
        sample['text_input'] = question
        sample['context'] = context

        sample = prepare_sample(sample, cuda_enabled=True)
        if(deterministic):
            torch.manual_seed(42)
        output = self.model.generate(sample, **kwargs)

        # output =  [t.replace(' ','') for t in output]

        return output
    
    def stream_generator(self, image, question, context, deterministic, kwargs):
        if image is not None:
            image = Image.open(image).convert('RGB')    
            image = self.vis_processors(image)

        question = [question]
        context = [context]

        sample = {}
        sample['image'] = torch.stack([image], dim=0)
        sample['text_input'] = question
        sample['context'] = context

        sample = prepare_sample(sample, cuda_enabled=True)
        if(deterministic):
            torch.manual_seed(42)
        stream = self.model.stream_generator(sample, **kwargs)

        return stream