# python -m scripts.write_out \
#     --output_base_path test \
#     --tasks mimic_iii_sum \
#     --sets test \
#     --num_fewshot 0 \
#     --num_examples 5 \
#     --description_dict_path test
# python main.py \
#     --model hf-causal \
#     --model_args pretrained=EleutherAI/pythia-160m \
#     --tasks mimic_cxr_sum \
#     --device cuda:0
# python main.py \
#     --model hf-causal \
#     --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
#     --tasks medqa_usmle \
#     --device cuda:0


import ast
from itertools import zip_longest

import datasets
import numpy as np
import torch.nn as nn
from f1chexbert import F1CheXbert
from lm_eval.base import rf, Task
from radgraph import F1RadGraph
from rouge_score import rouge_scorer


class F1RadGraphWrapper(F1RadGraph):
    def forward(self, items):
        preds = [item[0] for item in items]
        refs = [item[1] for item in items]
        score = super().forward(refs=refs, hyps=preds)
        return score[0]


class F1CheXbertWrapper(F1CheXbert):
    _instance = None
    _results_cache = None

    def __new__(cls, *args, **kwargs):
        # If the _instance does not exist, create it
        if not cls._instance:
            cls._instance = super(F1CheXbertWrapper, cls).__new__(cls)
        return cls._instance

    def forward(self, items, key):
        # If the results are already computed, return them
        if self._results_cache:
            return self._results_cache[key]

        preds = [item[0] for item in items]
        refs = [item[1] for item in items]
        accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = super().forward(refs=refs, hyps=preds)

        self._results_cache = {
            "chexbert-5_micro avg_f1-score": chexbert_5["micro avg"]["f1-score"],
            "chexbert-all_micro avg_f1-score": chexbert_all["micro avg"]["f1-score"],
            "chexbert-5_macro avg_f1-score": chexbert_5["macro avg"]["f1-score"],
            "chexbert-all_macro avg_f1-score": chexbert_all["macro avg"]["f1-score"]
        }

        return self._results_cache[key]


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
        preds = [item[0] for item in items]
        refs = [item[1] for item in items]
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
            test = datasets.arrow_dataset.Dataset.from_dict(self.dataset["test"][:2])
            return test
            return self.dataset["test"]

    def _process_doc(self, doc):
        return doc

    def construct_requests(self, doc, ctx):
        return [rf.greedy_until(ctx, {'until': ["\n"]})]

    def aggregation(self):
        chexbert_metrics = F1CheXbertWrapper()

        return {
            "rougeL": Rouge("rougeL"),
            "rouge1": Rouge("rouge1"),
            "rouge2": Rouge("rouge2"),
            "rougeL-Prec": Rouge("rougeL", measure="precision"),
            "rougeL-Rec": Rouge("rougeL", measure="recall"),
            "rougeL-F1": Rouge("rougeL", measure="fmeasure"),
            "F1RadGraph": F1RadGraphWrapper("simple"),
            "chexbert-5_micro avg_f1-score": lambda items: chexbert_metrics(items, "chexbert-5_micro avg_f1-score"),
            "chexbert-all_micro avg_f1-score": lambda items: chexbert_metrics(items,
                                                                                 "chexbert-all_micro avg_f1-score"),
            "chexbert-5_macro avg_f1-score": lambda items: chexbert_metrics(items, "chexbert-5_macro avg_f1-score"),
            "chexbert-all_macro avg_f1-score": lambda items: chexbert_metrics(items,
                                                                                 "chexbert-all_macro avg_f1-score")
        }

    def higher_is_better(self):
        return {
            "rougeL": True,
            "rouge1": True,
            "rouge2": True,
            "rougeL-Prec": True,
            "rougeL-Rec": True,
            "rougeL-F1": True,
            "F1RadGraph": True,
            "chexbert-5_micro avg_f1-score": True,
            "chexbert-all_micro avg_f1-score": True,
            "chexbert-5_macro avg_f1-score": True,
            "chexbert-all_macro avg_f1-score": True
        }


class MimicCXRSum(RadSum):
    VERSION = 0
    DATASET_PATH = "medarc/mimic-cxr-rrs"
    DATASET_NAME = None

    def doc_to_text(self, doc):
        return f"Summarize these radiology report findings to impression. Findings: {doc['findings']} Impression:"

    def doc_to_target(self, doc):
        return " " + doc['impression']

    def process_results(self, doc, results):
        return {
            "rouge1": (results[0].strip(), doc["impression"]),
            "rouge2": (results[0].strip(), doc["impression"]),
            "rougeL": (results[0].strip(), doc["impression"]),
            "F1RadGraph": (results[0].strip(), doc["impression"]),
            "chexbert-5_micro avg_f1-score": (results[0].strip(), doc["impression"]),
            "chexbert-all_micro avg_f1-score": (results[0].strip(), doc["impression"]),
            "chexbert-5_macro avg_f1-score": (results[0].strip(), doc["impression"]),
            "chexbert-all_macro avg_f1-score": (results[0].strip(), doc["impression"])
        }


class MimicIIISum(RadSum):
    VERSION = 0
    DATASET_PATH = "medarc/mimic-iii-rrs"
    DATASET_NAME = None

    def doc_to_text(self, doc):
        return f"Summarize these radiology report findings to impression. Findings: {doc['findings']} Impression:"

    def doc_to_target(self, doc):
        return " " + doc['impression']

    def process_results(self, doc, results):
        return {
            "rouge1": (results[0], doc["impression"]),
            "rouge2": (results[0], doc["impression"]),
            "rougeL": (results[0], doc["impression"]),
            "F1RadGraph": (results[0], doc["impression"])
        }


class ProblemListSum(RadSum):
    VERSION = 0
    DATASET_PATH = "medarc/problem_list_summarization"
    DATASET_NAME = None

    def doc_to_text(self, doc):
        return f"Summarize this Electronic Health Record Progress Note into " \
               f"Active Diagnoses and Problems (at most 10, comma separated). \n" \
               f"Progress Note: {doc['inputs']} \n" \
               f"Active Diagnoses and Problems:"

    def process_target(self, target):
        return ",".join(ast.literal_eval(target))

    def doc_to_target(self, doc):
        return " " + self.process_target(doc["target"])

    def process_results(self, doc, results):
        return {
            "rougeL-Prec": (results[0], self.process_target(doc["target"])),
            "rougeL-Rec": (results[0], self.process_target(doc["target"])),
            "rougeL-F1": (results[0], self.process_target(doc["target"]))
        }


class ConsumerHealthQuestion(RadSum):
    VERSION = 0
    DATASET_PATH = "medarc/consumer_health_questions"
    DATASET_NAME = None

    def doc_to_text(self, doc):
        return f"Summarize this consumer health questions into a condensed question expressing the minimum information " \
               f"required to find correct answers to the original question.\n" \
               f"Consumer health questions: {doc['inputs']}\n" \
               f"Condensed question:"

    def doc_to_target(self, doc):
        return " " + doc["target"]

    def process_results(self, doc, results):
        return {
            "rouge1": (results[0], doc["target"]),
            "rouge2": (results[0], doc["target"]),
            "rougeL": (results[0], doc["target"])
        }


class DialogueToNoteSum(RadSum):
    VERSION = 0
    DATASET_PATH = "medarc/Dialogue2Note_Summarization"
    DATASET_NAME = None

    def doc_to_text(self, doc):
        return f"Given the dialogue between a patient and a doctor, generate " \
               f"the Assessment and Plan section of a clinical note." \
               f"Dialogue: {doc['inputs']}\n" \
               f"Assessment and Plan section:"

    def has_validation_docs(self):
        """Whether the task has a validation set"""
        return False

    def doc_to_target(self, doc):
        return " " + doc["target"]

    def process_results(self, doc, results):
        return {
            "rouge1": (results[0], doc["target"]),
            "rouge2": (results[0], doc["target"]),
            "rougeL": (results[0], doc["target"])
        }
