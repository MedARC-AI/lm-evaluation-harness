import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Embeddings:
    def __init__(self, fewshot_col, task_description='', device='cuda') -> None:
        self.fewshot_col = fewshot_col
        self.task_description = task_description
        self.device = device
        self.post_init()

    def post_init(self):
        pass

    def __call__(self, doc):
        return self.get_embedding(doc)

    def get_embedding(self, doc):
        pass


class LookupEmbeddings(Embeddings):
    """
    Embedding is precomputed and should be located in self.fewshot_col + '_embed'
    """
    def get_embedding(self, doc):
        return doc[self.fewshot_col + '_embed']


class MistralEmbeddings(Embeddings):
    HF_MODEL = 'intfloat/e5-mistral-7b-instruct'
    MAX_LENGTH = 1024

    """
    Code largely copied from

    https://huggingface.co/intfloat/e5-mistral-7b-instruct

    SOTA at retrieval on https://huggingface.co/spaces/mteb/leaderboard as of 1/8/2024
    """

    def post_init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MistralEmbeddings.HF_MODEL)
        self.model = AutoModel.from_pretrained(MistralEmbeddings.HF_MODEL, device='auto').eval().to(self.device)

    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task_description}\nQuery: {query}'

    def get_embedding(self, doc):
        # Tokenize the input texts
        batch_dict = self.tokenizer(
            [self._get_detailed_instruct(self.task_description, doc[self.fewshot_col])],
            max_length=MistralEmbeddings.MAX_LENGTH - 1, return_attention_mask=False, padding=False, truncation=True
        )

        # append eos_token_id to every input_ids
        batch_dict['input_ids'] = [input_ids + [self.tokenizer.eos_token_id] for input_ids in batch_dict['input_ids']]
        batch_dict = self.tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
        batch_dict = {k: v.to(self.model.device) for k, v in batch_dict.items()}

        outputs = self.model(**batch_dict)
        embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings.cpu().detach().numpy().tolist()[0]


EMBEDDING_REGISTRY = {
    "mistral": MistralEmbeddings,
    'lookup': LookupEmbeddings,
}


def get_embedding_instance(name):
    if name is None:
        return LookupEmbeddings
    try:
        return EMBEDDING_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to use embedding '{name}', but no embedding for this name found! Supported model names: {', '.join(EMBEDDING_REGISTRY.keys())}"
        )
