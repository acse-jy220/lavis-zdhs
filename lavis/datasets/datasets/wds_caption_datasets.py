import webdataset as wds
from lavis.datasets.datasets.base_dataset import BaseDataset
import glob
import warnings
import random
import os

import re

def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    warnings.warn(repr(exn))
    return True

class WDSCaptionDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location, samples_per_inner_epoch):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        assert samples_per_inner_epoch > 0, "samples_per_inner_epoch must be greater than 0."

        tar_files = []
        for loc in location:
            files = glob.glob(loc+"/*.tar")
            select_files = []
            # if('laion_filtered_out' in loc):
            #     # 00000 - 09327.tar
            #     for f in files:
            #         if(int(f.split('/')[-1].split('.')[0]) <= 9327):
            #             select_files.append(f)

            #     tar_files.extend(select_files)
            #     continue
            if('cc12m' in loc):
                # 00000 - 00419.tar
                for f in files:
                    if(int(f.split('/')[-1].split('.')[0]) <= 419):
                        select_files.append(f)
                
                tar_files.extend(select_files)
                continue
            # if('cc3m' in loc):
            #     # 00000 - 00204.tar
            #     for f in files:
            #         if(int(f.split('/')[-1].split('.')[0]) <= 204):
            #             select_files.append(f)

            #     tar_files.extend(select_files)
            #     continue

            if('wukong_tar' in loc):
                # 00000 - 00305.tar 00678-00804.tar
                for f in files:
                    idx = int(f.split('/')[-1].split('.')[0])
                    if(idx <= 305 or (idx >= 678 and idx <= 804)):
                        select_files.append(f)

                tar_files.extend(select_files)
                continue
            
            tar_files.extend(files)
                
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(tar_files,deterministic=True),
            wds.tarfile_to_samples(handler=warn_and_continue),
            wds.shuffle(1000, handler=warn_and_continue),
            wds.decode("pilrgb", handler=warn_and_continue),
            wds.to_tuple("jpg", "txt", "__url__", handler=warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=warn_and_continue),
            wds.map(self.to_dict, handler=warn_and_continue),
        ).with_epoch(samples_per_inner_epoch)

        # self.inner_dataset = wds.DataPipeline(
        #     wds.ResampledShards(tar_files,deterministic=True),
        #     wds.tarfile_to_samples(handler=wds.ignore_and_continue),
        #     wds.shuffle(1000, handler=wds.ignore_and_continue),
        #     wds.decode("pilrgb", handler=wds.ignore_and_continue),
        #     wds.to_tuple("jpg", "txt", handler=wds.ignore_and_continue),
        #     wds.map_tuple(self.vis_processor, handler=wds.ignore_and_continue),
        #     wds.map(self.to_dict, handler=wds.ignore_and_continue),
        # ).with_epoch(samples_per_inner_epoch)

    # def to_dict(self, sample):
    #     lang = '图片标题'
    #     if(('laion_filtered_out' in sample[2]) or ('cc12m-filter' in sample[2]) or ('cc3m-filter' in sample[2])  or ('coco-en-filter' in sample[2]) ):
    #         lang = 'image caption'
    #     return {
    #         "image": sample[0],
    #         "text_input": self.text_processor(sample[1]),
    #         "lang": lang
    #     }
    
    def to_dict(self, sample):
        lang = '图片标题'
        if(('laion_filtered_out' in sample[2]) or ('cc12m-filter' in sample[2]) or ('cc3m-filter' in sample[2])  or ('coco-en-filter' in sample[2]) or ('coyo-700m-cn_clip' in sample[2])):
            lang = 'image caption'
        return {
            "image": sample[0],
            "text_output": self.text_processor(sample[1]),
            "text_input": lang,
            "context": "",
        }