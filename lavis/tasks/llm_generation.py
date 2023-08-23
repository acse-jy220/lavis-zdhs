"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

# from lavis.tool.caption.pycxevalcap.eval import COCOEvalCap
# from lavis.tool.caption.pycxtools.coco import COCO

@registry.register_task("llm_generation")
class LLMGenerationTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, is_caption = True, gt_file=None, candidates_file=None, n_segments=1, eval_batch=1):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.is_caption = is_caption
        self.gt_file = gt_file
        self.candidates_file = candidates_file
        if candidates_file is not None:
            self.candidates = json.load(open(self.candidates_file))
        self.n_segments = n_segments
        self.cur_candidate = 0
        self.eval_batch = eval_batch

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate
        is_caption = run_cfg.get("is_caption", True)
        gt_file = run_cfg.get("gt_file", None)
        candidates_file = run_cfg.get("candidates_file", None)
        n_segments = run_cfg.get("n_segments", 1)
        eval_batch = run_cfg.get("batch_size_eval", 1)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            is_caption=is_caption,
            gt_file=gt_file,
            candidates_file=candidates_file,
            n_segments=n_segments,
            eval_batch=eval_batch,
        )

    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        if self.is_caption:
            captions = model.generate(
                samples,
                use_nucleus_sampling=False,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
                max_new_tokens=self.max_len,
            )
        else:
            captions = model.predict_class(
                samples,
                self.candidates[self.cur_candidate:self.cur_candidate+self.eval_batch],
                self.n_segments
            )
            self.cur_candidate = self.cur_candidate+self.eval_batch

        img_ids = samples["image_id"]
        for caption, img_id in zip(captions, img_ids):
            results.append({"caption": caption, "image_id": img_id})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )


#     @main_process
#     def _report_metrics(self, eval_result_file, split_name):

#         if self.is_caption:
#             coco_val = coco_caption_eval(self.coco, eval_result_file)

#             agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
#             log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

#             eval_res = {k: v for k, v in coco_val.eval.items()}
#             eval_res["agg_metrics"] = agg_metrics
#         else:
#             qa_val = qa_accuracy_eval(self.gt_file, eval_result_file)
#             log_stats = {split_name: {k: v for k, v in qa_val.eval.items()}}
#             eval_res = {k: v for k, v in qa_val.items()}


#         with open(
#             os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
#         ) as f:
#             f.write(json.dumps(log_stats) + "\n")

#         return eval_res

# def qa_accuracy_eval(qa_gt, results_file):

#     def processAns(ans):
#         ans = ans.replace('\n', ' ')
#         ans = ans.replace('\t', ' ')
#         ans = ans.strip()
#         return ans

#     gts = {}
#     gts_l = json.load(open(qa_gt))
#     for ins in gts_l:
#         ques = ins['image_id']
#         ans = ins['caption']
#         gts[ques] = processAns(ans)
#     res = {}
#     res_l = json.load(open(results_file))
#     for ins in res_l:
#         ques = ins['image_id']
#         ans = ins['caption']
#         res[ques] = processAns(ans)

#     sample = 0
#     match = 0
#     for key, val in res.items():
#         sample = sample + 1
#         match = match + (1 if val == gts[key] else 0)
    
#     return {'accuracy': round(match * 100.0 / sample,2)}
