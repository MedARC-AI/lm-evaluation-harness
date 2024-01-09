import argparse
from datasets import DatasetDict
from openai import AzureOpenAI
import numpy as np
np.random.seed(1992)
from textwrap import dedent
import regex as re

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.models.openai_completions import OpenaiChatCompletionsLM
from lm_eval.tasks.pubmedqa.mistral_embeddings import initialize_embedding_model, get_question_embedding


DATASET_TO_INPUT_TYPE = {
    'pubmedqa_medprompt': 'abstract',
    'medmcqa_fewshot_cot': None,
}


DATASET_TO_CONTEXTS_COL = {
    'pubmedqa_medprompt': 'CONTEXTS',
    'medmcqa_fewshot_cot': None,
}


DATASET_TO_QUESTION_COL = {
    'pubmedqa_medprompt': 'QUESTION',
    'medmcqa_fewshot_cot': 'question',
}


def chatgpt(client, messages, model='gpt-4', temperature=0.1, max_tokens=2048):
    completion = client.with_options(max_retries=5).chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
    )
    return completion.choices[0].message.content


def build_prompt(doc, task):
    """
    Prompt is taken from MedPrompt
    https://github.com/microsoft/promptbase/blob/90fe3f1e2476638ae7e623687bfe9b8b2077b2bb/src/promptbase/drop/drop.py#L98
    """

    choice_str = ', '.join(task.doc_to_choice(doc))
    question = doc[DATASET_TO_QUESTION_COL[task.config.task]]

    if 'pubmed' in task.config.task:
        input = '\n'.join(doc[DATASET_TO_CONTEXTS_COL[task.config.task]])
        input_type = DATASET_TO_INPUT_TYPE[task.config.task]

        prompt = dedent(f"""
        Answer the following reading comprehension **Question** based on the **{input_type}** below.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the **Final Answer** from the choice set: {choice_str}.
        ----
        **{input_type}:** {input}
        ----
        **Question:** {question}
        ----
        **Explanation**: """
        )
    else:
        choice_letters = ['A', 'B', 'C', 'D']
        choice_options = [
            doc['opa'],
            doc['opb'],
            doc['opc'],
            doc['opd'],
        ]

        choice_str = []
        for l, o in zip(choice_letters, choice_options):
            choice_str.append(f'{l}) {o}')
        choice_str = '\n'.join(choice_str)
        choice_letter_str = ', '.join(choice_letters)

        prompt = dedent(f"""
        Answer the following reading comprehension **Question**.
        First, think step by step and write an **Explanation** for reasoning through the question.
        Then, analyze your explanation and write just the Letter ({choice_letter_str}) corresponding to your **Final Answer**.
        ----s
        **Question:** {question}
        ----
        **Choices:**\n{choice_str}\n
        ----
        **Explanation**: """
        )

    prompt = '\n'.join([x.lstrip() for x in prompt.split('\n')])

    return prompt

def generate_self_cot(doc, task, lm_obj, embeddings, add_self_cot=True, consistency_filter=True):
    question = doc[DATASET_TO_QUESTION_COL[task.config.task]]
    q_embed = get_question_embedding(question, embeddings['model'], embeddings['tokenizer'])

    new_cols = {f'{DATASET_TO_QUESTION_COL[task.config.task]}_embed': q_embed, 'rationale': ''}
    if not add_self_cot:  # We don't pre-compute CoT for every split. Only "fewshot_split"
        return new_cols

    prompt = build_prompt(doc, task)

    # Empty gen config for now
    args = (prompt, {})

    instance = lm_eval.api.instance.Instance(
        request_type='generate_until',
        arguments=args,
        doc=doc,
        idx=0
    )

    if type(lm_obj) == AzureOpenAI:
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant for medical question answering.'},
            {'role': 'user', 'content': prompt}
        ]
        prediction = chatgpt(client=lm_obj, messages=messages)
    else:
        prediction = lm_obj.generate_until(requests=[instance])[0]

    if type(prediction) != str:
        print(f'Model returned prediction of type {type(prediction)} -> {prediction}. Returning with no CoT')
        return new_cols

    split = re.split(re.escape('**Final Answer'), prediction)
    if len(split) != 2:
        print('Invalid output format. Check your prompt.')
        return new_cols

    rationale, answer_raw = split
    rationale = rationale.strip()

    choice_regex = r'|'.join(task.doc_to_choice(doc))
    answer_match = re.search(rf'({choice_regex})', answer_raw, flags=re.IGNORECASE)
    if answer_match is None:
        print(f'Expected one of {choice_regex}. Got {answer_raw}. Check your prompt.')
        return new_cols

    answer_lower = answer_match.group().lower()
    target = task.doc_to_target(doc)
    if type(target) == int:
        target = task.doc_to_choice(doc)[target]
    assert target in task.doc_to_choice(doc)
    target_lower = target.lower()

    if consistency_filter and answer_lower != target_lower:
        print(f'Answer ({answer_lower}) didn\'t match ground truth target ({target_lower}). Not adding to CoT dataset.')
        return new_cols

    new_cols.update({'rationale': rationale})

    return new_cols


if __name__ == '__main__':
    # generate and cache self-COT examplars for few-shot https://github.com/microsoft/promptbase
    parser = argparse.ArgumentParser('Pre-Compute CoT rationales for PubMedQA')

    parser.add_argument('--task', default='pubmedqa_fewshot_cot')
    parser.add_argument('--fewshot_split', default='validation', choices=['training', 'validation', 'test'])
    parser.add_argument('--max_cot_examples', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--device', default=0, type=int)

    parser.add_argument('--cot_model', default='gpt-4')
    parser.add_argument('--cot_model_type', default='openai', choices=['openai', 'huggingface'])
    parser.add_argument('-remove_consistency_filter', default=False, action='store_true')

    args = parser.parse_args()

    output_splits = {}
    args.save_dir = f'~/.cache/huggingface/datasets/{args.task}_{args.cot_model}_preprocessed'
    args.hub_dir = f'medarc/{args.task}_{args.cot_model}_preprocessed'

    lm_eval.tasks.initialize_tasks()

    # Get Task object
    task = lm_eval.tasks.get_task_dict(args.task)[args.task]

    print('Initializing embeddings...')
    embeddings = initialize_embedding_model(args.device)

    # Download the task from HF or HF cache
    task.download()

    for split in ['training', 'validation', 'test']:
        add_self_cot = split == args.fewshot_split
        assert getattr(task, f'has_{split}_docs')()
        cot_data = getattr(task, f'{split}_docs')()
        n = len(cot_data)
        if add_self_cot and args.max_cot_examples < n:
            data_idxs = np.arange(n)
            np.random.shuffle(data_idxs)
            sample_idxs = data_idxs[:args.max_cot_examples]
            # TODO: Can shuffle first for randomization but this is just for debugging purposes
            print(f'Randomly sampling {args.max_cot_examples} examples from {n} total...')
            cot_data = cot_data.select(sample_idxs)

        if args.cot_model_type == 'openai':
            # lm_obj = OpenaiChatCompletionsLM(
            #     model=args.cot_model,
            # )
            import os
            assert 'OPENAI_API_KEY' in os.environ
            lm_obj = AzureOpenAI(
                api_key=os.environ.get('OPENAI_API_KEY'),
                azure_endpoint='https://east-us-2-llm.openai.azure.com/',
                api_version='2023-05-15',
                azure_deployment='misc-gpt4-turbo'
            )
        else:
            lm_obj = HFLM(
                pretrained=args.cot_model,
                device='cuda:0',
                batch_size=args.batch_size,
            )

        cot_data_w_cot = cot_data.map(
            lambda doc: generate_self_cot(
                doc, task, lm_obj, embeddings, add_self_cot=add_self_cot,
                consistency_filter=not args.remove_consistency_filter
            )
        )

        cot_data_w_cot_valid = cot_data_w_cot.filter(lambda example: not add_self_cot or len(example['rationale']) > 0)
        valid_n = len(cot_data_w_cot_valid)

        print(f'Recorded {valid_n} / {n} CoT examples for {split} split')
        output_splits[split] = cot_data_w_cot_valid

    print(f'Saving to {args.save_dir} and to {args.hub_dir}...')
    dataset = DatasetDict(output_splits)
    dataset.save_to_disk(args.save_dir)
    dataset.push_to_hub(args.hub_dir, private=True)
