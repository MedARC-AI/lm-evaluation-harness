# TODO: Remove all TODO comments once the implementation is complete.
"""
What disease does this patient have? a large-scale open domain question answering dataset from medical exams
https://arxiv.org/abs/2009.13081

In this work, we present the first free-form multiple-choice OpenQA dataset for solving medical problems, MedQA,
collected from the professional medical board exams. It covers three languages: English, simplified Chinese, and
traditional Chinese, and contains 12,723, 34,251, and 14,123 questions for the three languages, respectively. Together
with the question data, we also collect and release a large-scale corpus from medical textbooks from which the reading
comprehension models can obtain necessary knowledge for answering the questions.

Homepage: https://github.com/jind11/MedQA
"""
from lm_eval.base import MultipleChoiceTask


# TODO: Add the BibTeX citation for the task.
_CITATION = """
@article{jin2021disease,
  title={What disease does this patient have? a large-scale open domain question answering dataset from medical exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={Applied Sciences},
  volume={11},
  number={14},
  pages={6421},
  year={2021},
  publisher={MDPI}
}
"""


class MedQA4Options(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "GBaker/MedQA-USMLE-4-options-hf"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            # We cache training documents in `self._training_docs` for faster
            # few-shot processing. If the data is too large to fit in memory,
            # return the training data as a generator instead of a list.
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        return {
            "query": doc["sent1"],  # The query prompt.
            "choices": [doc["ending0"], doc["ending1"], doc["ending2"], doc["ending3"]],  # The list of choices.
            "gold": doc["label"],  # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        option_choices = {'A':doc["choices"][0], 'B':doc["choices"][1], 'C':doc["choices"][2], 'D':doc["choices"][3]}
        answers = "".join((f"({k}) {v}\n") for k,v in option_choices.items())
        return f"Question: {doc['query']}\n{answers}Answer:"
