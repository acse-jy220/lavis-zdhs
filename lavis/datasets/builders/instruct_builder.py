from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.instruct_datasets import (
    InstructDataset,
    InstructEvalDataset,
)

from lavis.common.registry import registry

@registry.register_builder("zdtc_blip2_instruct")
class InstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = InstructDataset
    eval_dataset_cls = InstructEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/zdtc/defualts_blip2_instruct.yaml",
    }

    def _download_data(self):
        self._download_vis()