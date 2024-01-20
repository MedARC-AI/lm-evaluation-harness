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


def doc_to_choice(doc, return_letters=True):
    return ["A", "B", "C"] if return_letters else CHOICES


def shuffled_choice_list(letters, options, shuffle=True):
    n = len(letters)
    order = np.arange(n)
    if shuffle:
        np.random.shuffle(order)

    options_shuffled = [options[i] for i in order]
    unshuffle_map = {letters[new_idx]: letters[old_idx] for new_idx, old_idx in enumerate(order)}

    def unshuffle_answer_callback(output):
        pattern = r'<{0,2}Final Answer:?>{0,2}:?\s?([A-C])'
        match = re.search(pattern, output, flags=re.IGNORECASE)
        if match is None:
            pattern = r'answer is\W*([A-C])'
            match = re.search(pattern, output, flags=re.IGNORECASE)
            if match is None:
                option_str = '|'.join(list(map(re.escape, options)))
                literal_pattern = r'<{0,2}Final Answer:?>{0,2}:?\s?(' + option_str + ')'
                match = re.search(literal_pattern, output, flags=re.IGNORECASE)
                if match is None:
                    raise Exception(f'Answer Not Found! Check output below.\n{output}')
                else:
                    option_idx = options.index(match.group(1).strip())
                    return output[:match.start()] + '<<Final Answer:>> ' + letters[option_idx]
        return output[:match.start()] + '<<Final Answer:>> ' + unshuffle_map[match.group(1).strip().upper()]

    shuffled_str = '\n'.join([f'{l}) {c}' for l, c in zip(letters, options_shuffled)])
    return f'<<Choices:>>\n{shuffled_str}\n----\n<<Explanation:>>', unshuffle_answer_callback


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
