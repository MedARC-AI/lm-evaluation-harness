import torch.nn as nn
from rouge_score import rouge_scorer
from itertools import zip_longest
import numpy as np
from radgraph import F1RadGraph
from f1chexbert import F1CheXbert


def rouge1(predictions, references): return predictions[0], references[0]


def rouge2(predictions, references): return predictions[0], references[0]


def rougeL(predictions, references): return predictions[0], references[0]


def RadGraphF1(predictions, references): return predictions[0], references[0]


def chexbert_five_micro_avg_f1_score(predictions, references): return predictions[0], references[0]


def chexbert_all_micro_avg_f1_score(predictions, references): return predictions[0], references[0]


def chexbert_five_macro_avg_f1_score(predictions, references): return predictions[0], references[0]


def chexbert_all_macro_avg_f1_score(predictions, references): return predictions[0], references[0]


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


def AggRougeLPrec(items):
    return Rouge("rougeL", measure="precision")(items)


def AggRougeLRec(items):
    return Rouge("rougeL", measure="recall")(items)


def AggRougeLF1(items):
    return Rouge("rougeL", measure="fmeasure")(items)


def AggRougeL(items):
    return Rouge("rougeL")(items)


def AggRouge1(items):
    return Rouge("rouge1")(items)


def AggRouge2(items):
    return Rouge("rouge2")(items)


class F1RadGraphWrapper(F1RadGraph):
    def forward(self, items):
        preds = [item[0] for item in items]
        refs = [item[1] for item in items]
        score = super().forward(refs=refs, hyps=preds)
        return score[0]


def AggF1RadGraph(items):
    return F1RadGraphWrapper("simple")(items)


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


def agg_chexbert_five_micro_avg_f1_score(items):
    m = F1CheXbertWrapper()
    return m(items, "chexbert-5_micro avg_f1-score")


def agg_chexbert_all_micro_avg_f1_score(items):
    m = F1CheXbertWrapper()
    return m(items, "chexbert-all_micro avg_f1-score")


def agg_chexbert_five_macro_avg_f1_score(items):
    m = F1CheXbertWrapper()
    return m(items, "chexbert-5_macro avg_f1-score")


def agg_chexbert_all_macro_avg_f1_score(items):
    m = F1CheXbertWrapper()
    return m(items, "chexbert-all_macro avg_f1-score")
