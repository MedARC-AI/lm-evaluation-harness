import regex as re

import numpy as np
np.random.seed(1992)


LETTER_OPTIONS = ['A', 'B', 'C', 'D']


def doc_to_text(doc) -> str:
    """
    Question: <question>
    Choices:
    A. <choice1>
    B. <choice2>
    C. <choice3>
    D. <choice4>
    Answer:
    """
    option_choices = {l: doc[f'op{l.lower()}'] for l in LETTER_OPTIONS}

    prompt = "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt


def doc_to_choice(doc, return_letters=True):
    return LETTER_OPTIONS if return_letters else [doc[f'op{l.lower()}'] for l in LETTER_OPTIONS]


def doc_to_text_medprompt(doc):
    question = doc['question']
    text = f'<<Question:>> {question}\n----\n'
    return text


def doc_to_fewshot_text(doc):
    question = doc['question']
    # Include rationale if in dataset (this means either that it was pre-computed or is part of fewshot context)
    explanation_str = doc.get('RATIONALE', '')
    choice_str = '\n'.join([f"{l}) {doc['op'] + l.lower()}" for l in LETTER_OPTIONS])
    text = f'<<Question:>> {question}\n----\n<<Choices:>>\n{choice_str}\n----\n<<Explanation:>> {explanation_str}\n----\n<<Final Answer:>>'
    return text


def letter_target(doc):
    return LETTER_OPTIONS[doc['cop']]
