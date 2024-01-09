"""
Borrowed from

https://huggingface.co/intfloat/e5-mistral-7b-instruct

SOTA at retrieval on https://huggingface.co/spaces/mteb/leaderboard as of 1/8/2024
"""

import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


HF_MODEL = 'intfloat/e5-mistral-7b-instruct'
MAX_LENGTH = 1024
TASK_DESCRIPTION = 'Given a medical question, retrieve related medical questions'


def initialize_embedding_model(device):
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    model = AutoModel.from_pretrained(HF_MODEL).eval().to(device)
    return {
        'model': model,
        'tokenizer': tokenizer
    }


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def _get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def get_question_embedding(question, model, tokenizer):
    # Tokenize the input texts
    batch_dict = tokenizer(
        [_get_detailed_instruct(TASK_DESCRIPTION, question)],
        max_length=MAX_LENGTH - 1, return_attention_mask=False, padding=False, truncation=True
    )

    # append eos_token_id to every input_ids
    batch_dict['input_ids'] = [input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
    batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
    batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

    outputs = model(**batch_dict)
    embeddings = _last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings.cpu().detach().numpy().tolist()[0]
