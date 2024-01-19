import numpy as np


from lm_eval.api.embeddings import get_embedding_instance


class ContextSampler:
    def __init__(
            self, docs, task, fewshot_indices=None, rnd=None, fewshot_embedding_col=None, fewshot_embedding_model=None,
            fewshot_embedding_task_description=None,
    ) -> None:
        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"

        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_fewshot_text = self.task.doc_to_fewshot_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.fewshot_embedding_col = fewshot_embedding_col
        if fewshot_embedding_col is not None:
            self.fewshot_embedder = get_embedding_instance(
                fewshot_embedding_col, fewshot_embedding_model, fewshot_embedding_task_description
            )

        self.docs = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            self.docs = self.docs.select(fewshot_indices)

    def get_context(self, doc, num_fewshot):
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = (
            num_fewshot + 1
            if self.config.fewshot_split == self.config.test_split
            else num_fewshot
        )

        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(doc, n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]

        labeled_examples = (
            self.fewshot_delimiter.join(
                [
                    # TODO: is separating doc_to_text and doc_to_target by one space always desired?
                    (
                        self.doc_to_fewshot_text(doc)
                        if (
                            self.config.doc_to_choice is None
                            or type(self.doc_to_fewshot_text(doc)) is str
                        )
                        else self.doc_to_choice(doc)[self.doc_to_fewshot_text(doc)]
                    )
                    + self.target_delimiter
                    + (
                        str(self.doc_to_target(doc)[0])
                        if type(self.doc_to_target(doc)) is list
                        else self.doc_to_target(doc)
                        if (
                            self.config.doc_to_choice is None
                            or type(self.doc_to_target(doc)) is str
                        )
                        else str(self.doc_to_choice(doc)[self.doc_to_target(doc)])
                    )
                    for doc in selected_docs
                ]
            )
            + self.fewshot_delimiter
        )

        return labeled_examples

    def sample(self, doc, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.rnd.sample(self.docs, n)


class FirstNSampler(ContextSampler):
    def sample(self, doc, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(
            self.docs
        ), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, doc, n) -> None:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(ContextSampler):
    def sample(self, doc, n) -> None:
        """ """
        pass


class NearestNeighborsSampler(ContextSampler):
    def cosine_similarity(self, embed, fewshot_embeds):
        """Calculate the cosine similarity between two vectors."""
        dot_product = np.dot(embed, fewshot_embeds.transpose(1, 0))
        norm_v1 = np.linalg.norm(embed)
        norm_v2 = np.linalg.norm(fewshot_embeds)
        return dot_product / (norm_v1 * norm_v2)

    def sample(self, doc, n) -> None:
        """
        Use the embedding `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(
            self.docs
        ), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."

        q_embed = np.array(get_question_embedding(
            doc[self.fewshot_embedding_col], self.embedder['model'], self.embedder['tokenizer']
        ))

        # TODO: Should all have this column so take this out once preprocess_pubmedqa is run.
        default = np.random.random((4096, ))
        fewshot_embeds = np.array(list(map(
            lambda doc: doc.get(self.fewshot_embedding_col + '_embed', default), self.docs
        )))

        sims = self.cosine_similarity(q_embed, fewshot_embeds)
        top_indices = np.argsort(-sims)[:n]

        return list(map(lambda idx: self.docs[idx], top_indices))


SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
    "nearest_neighbors": NearestNeighborsSampler,
}


def get_sampler(name):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        )
