# Copied from Master
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
    choices = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]
    option_choices = {'A': choices[0], 'B': choices[1], 'C': choices[2], 'D': choices[3]}

    prompt = "Question: " + doc["question"] + "\nChoices:\n"
    for choice, option in option_choices.items():
        prompt += f"{choice.upper()}. {option}\n"
    prompt += "Answer:"
    return prompt

import regex as re

import numpy as np
np.random.seed(1992)


# TODO: this must match the doc_to_choice in task.config (dangerous to have twice?)
CHOICES = ['yes', 'no', 'maybe']


def doc_to_text(doc) -> str:
    ctxs = "\n".join(doc["CONTEXTS"])
    return "Abstract: {}\nQuestion: {}\nAnswer:".format(ctxs, doc["QUESTION"])


def letter_target(doc) -> int:
    return CHOICES.index(doc['final_decision'])


def shuffled_choice_list(shuffle=True):
    order = np.arange(len(CHOICES))
    if shuffle:
        np.random.shuffle(order)

    choices_shuffled = [CHOICES[i] for i in order]
    letters = ['A', 'B', 'C']
    unshuffle_map = {letters[new_idx]: letters[old_idx] for new_idx, old_idx in enumerate(order)}

    def unshuffle_answer_callback(output):
        pattern = r'<{0,2}Final Answer:?>{0,2}:?\s?([A-C])'
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match is None:
            pattern = r'answer is\W*([A-C])'
            match = re.search(pattern, output, flags=re.IGNORECASE)
            if match is None:
                raise Exception(f'Answer Not Found! Check output below.\n{output}')

        return output[:match.start()] + '<<Final Answer:>> ' + unshuffle_map[match.group(1).strip().upper()]

    choices_shuffled_str = '\n'.join([f'{l}) {c}' for l, c in zip(letters, choices_shuffled)])
    return "<<Choices:>>\n{}\n----\n<<Explanation:>>".format(choices_shuffled_str), unshuffle_answer_callback


def doc_to_text_cot(doc):
    ctxs = "\n".join(doc["CONTEXTS"])
    question = doc["QUESTION"]
    text = f'<<Abstract:>> {ctxs}\n----\n<<Question:>> {question}\n----\n<<Explanation:>>'
    return text


def doc_to_text_fewshot_cot(doc):
    ctxs = "\n".join(doc["CONTEXTS"])
    question = doc["QUESTION"]
    # Include rationale if in dataset (this means either that it was pre-computed or is part of fewshot context)
    explanation_str = doc.get("RATIONALE", "")

    letters = ['A', 'B', 'C']
    choice_str = '\n'.join([f'{l}) {c}' for l, c in zip(letters, CHOICES)])

    text = f'<<Abstract:>> {ctxs}\n----\n<<Question:>> {question}\n----\n<<Choices:>>\n{choice_str}\n----\n<<Explanation:>> {explanation_str}\n----\n<<Final Answer:>>'

    return text

