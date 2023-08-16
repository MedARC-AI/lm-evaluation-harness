# python -m scripts.write_out \
#     --output_base_path test \
#     --tasks mimic_iii_sum \
#     --sets test \
#     --num_fewshot 0 \
#     --num_examples 5 \
#     --description_dict_path test
from abc import ABC
# python main.py \
#     --model hf-causal \
#     --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
#     --tasks mimic_iii_sum \
#     --device cuda:0


from itertools import zip_longest

from lm_eval.base import rf, Task
from rouge_score import rouge_scorer
import numpy as np
import torch.nn as nn
import datasets


class Rouge(nn.Module):
    def __init__(self, rouges, measure="fmeasure", **kwargs):
        super().__init__()
        assert type(rouges) == str or type(rouges) == list
        assert measure in ["fmeasure", "precision", "recall"]
        if type(rouges) == str:
            rouges = [rouges]

        rouges = [r.replace('rougel', 'rougeL') for r in rouges]
        self.scorer = rouge_scorer.RougeScorer(rouges, use_stemmer=True)
        self.rouges = rouges
        self.measure = measure

    def forward(self, items):
        preds = list(zip(*items))[0]
        refs = list(zip(*items))[1]
        scores = []
        for target_rec, prediction_rec in zip_longest(refs, preds):
            if target_rec is None or prediction_rec is None:
                raise ValueError("Must have equal number of lines across target and "
                                 "prediction.")
            scores.append(self.scorer.score(target_rec, prediction_rec))
        f1_rouge = [getattr(s[self.rouges[0]], self.measure) for s in scores]
        return np.mean(f1_rouge)


# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""


class RadSum(Task):
    def __init__(self, **kwargs):
        self.dataset = None
        super().__init__(**kwargs)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    def has_training_docs(self):
        """Whether the task has a training set"""
        return True

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return True

    def has_test_docs(self):
        """Whether the task has a test set"""
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def _process_doc(self, doc):
        return doc

    def construct_requests(self, doc, ctx):
        return [rf.greedy_until(ctx, {'until': ["\n"]})]

    def aggregation(self):
        return {
            "rougeL": Rouge("rougeL"),
            "rouge1": Rouge("rouge1"),
            "rouge2": Rouge("rouge2"),
        }

    def higher_is_better(self):
        return {
            "rougeL": True,
            "rouge1": True,
            "rouge2": True,
        }


class MimicIIISum(RadSum):
    VERSION = 0
    DATASET_PATH = "medarc/mimic-iii-rrs"
    DATASET_NAME = None

    def doc_to_text(self, doc):
        return f"Summarize these radiology report findings to impression. Findings:{doc['findings']} Impression:"

    def doc_to_target(self, doc):
        return " " + doc['impression']

    def process_results(self, doc, results):
        return {
            "rouge1": (results, doc["impression"]),
            "rouge2": (results, doc["impression"]),
            "rougeL": (results, doc["impression"])
        }
