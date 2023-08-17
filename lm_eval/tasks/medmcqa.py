"""
MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering
https://proceedings.mlr.press/v174/pal22a/pal22a.pdf

MedMCQA is a large-scale, Multiple-Choice Question Answering (MCQA) dataset 
designed to address real-world medical entrance exam questions. MedMCQA has more
than 194k high-quality AIIMS & NEET PG entrance exam MCQs covering 2.4k 
healthcare topics and 21 medical subjects are collected with an average token 
length of 12.77 and high topical diversity. Each sample contains a question, 
correct answer(s), and other options which require a deeper language 
understanding as it tests the 10+ reasoning abilities of a model across a wide 
range of medical subjects & topics. A detailed explanation of the solution, 
along with the above information, is provided in this study.

Homepage: https://medmcqa.github.io/
"""
from lm_eval.base import MultipleChoiceTask


_CITATION = """
@InProceedings{pmlr-v174-pal22a,
    title = 	 {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
    author =       {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
    booktitle = 	 {Proceedings of the Conference on Health, Inference, and Learning},
    pages = 	 {248--260},
    year = 	 {2022},
    editor = 	 {Flores, Gerardo and Chen, George H and Pollard, Tom and Ho, Joyce C and Naumann, Tristan},
    volume = 	 {174},
    series = 	 {Proceedings of Machine Learning Research},
    month = 	 {07--08 Apr},
    publisher =    {PMLR},
    pdf = 	 {https://proceedings.mlr.press/v174/pal22a/pal22a.pdf},
    url = 	 {https://proceedings.mlr.press/v174/pal22a.html},
    abstract = 	 {This paper introduces MedMCQA, a new large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. More than 194k high-quality AIIMS & NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected with an average token length of 12.77 and high topical diversity. Each sample contains a question, correct answer(s), and other options which requires a deeper language understanding as it tests the 10+ reasoning abilities of a model across a wide range of medical subjects & topics. A detailed explanation of the solution, along with the above information, is provided in this study.}
}
"""


class MedMCQA(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "medmcqa"
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
            "query": doc["question"],  # The query prompt.
            "choices": [doc["opa"], doc["opb"], doc["opc"], doc["opd"]],  # The list of choices.
            "gold": doc["cop"],  # The integer used to index into the correct element of `"choices"`.
        }

    def doc_to_text(self, doc):
        prompt = "The following are multiple choice questions (with answers) about medical knowledge."
        option_choices = {'A':doc["choices"][0], 'B':doc["choices"][1], 'C':doc["choices"][2], 'D':doc["choices"][3]}
        answers = "".join((f"({k}) {v}\n") for k,v in option_choices.items())
        return f"{prompt}\nQuestion: {doc['query']}\n{answers}\nAnswer:"
