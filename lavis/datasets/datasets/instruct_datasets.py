"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from PIL import Image
import random

from lavis.datasets.datasets.base_dataset import BaseDataset

from collections import OrderedDict
from transformers.utils import logging

logger = logging.get_logger(__name__)

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "answer": ann["answer"],
                "image": sample["image"],
                "image_id": ann['image_id'],
            }
        )


class InstructDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # self.img_ids = {}
        # n = 0
        # for ann in self.annotation:
        #     img_id = ann["image_id"]
        #     if img_id not in self.img_ids.keys():
        #         self.img_ids[img_id] = n
        #         n += 1
        self.length = len(self.annotation)

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        while(1):
            try:
                ann = self.annotation[index]
                image_path = os.path.join(self.vis_root, ann["image"])
                image = Image.open(image_path).convert("RGB")
                break
            except FileNotFoundError as e:
                logger.warn(
                        "FileNotFound: {}".format(image_path)
                    )
                index = random.randint(0,self.length-1)

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        answer = self.text_processor(ann["answer"])
        context = self.text_processor(ann.get("context",""))

        return {
            "image": image,
            "text_input": question,
            "text_output": answer,
            "context": context,
        }


class InstructEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        context = self.text_processor(ann.get("context",""))

        return {
            "image": image,
            "text_input": question,
            "image_id": ann['image_id'],
            "context": context,
        }