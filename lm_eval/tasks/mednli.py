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
import datasets

from lm_eval.base import MultipleChoiceTask

# TODO: Add the BibTeX citation for the task.
_CITATION = """
@misc{https://doi.org/10.13026/c2rs98,
    title        = {MedNLI — A Natural Language Inference Dataset For The Clinical Domain},
    author       = {Shivade,  Chaitanya},
    year         = 2017,
    publisher    = {physionet.org},
    doi          = {10.13026/C2RS98},
    url          = {https://physionet.org/content/mednli/}
}
"""


class MedNLI(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "bigbio/mednli"
    DATASET_NAME = "mednli_bigbio_te"

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None):
        super().__init__("lm_eval/datasets/mednli", cache_dir, download_mode)

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

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
        def _format_question(doc):
            premise = doc['premise'].strip()
            premise = premise if premise[-1:] == '.' else premise + '.'
            hypothesis = doc['hypothesis'].strip()
            hypothesis = hypothesis if hypothesis[-1:] == '.' else hypothesis + '.'
            return f"Answer entailment, contradiction or neutral. Premise: {premise} Hypothesis: {hypothesis}"

        return {
            "query": _format_question(doc),
            "choices": ['entailment', 'contradiction', 'neutral'],
            "gold": ['entailment', 'contradiction', 'neutral'].index(doc["label"]),
        }

    def doc_to_text(self, doc):
        return doc['query']
