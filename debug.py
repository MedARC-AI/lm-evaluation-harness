import argparse

import lm_eval
from lm_eval.models.huggingface import HFLM


DEBUG_MODEL = 'HuggingFaceM4/tiny-random-LlamaForCausalLM'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Debug for MedPrompt')

    parser.add_argument('--model', default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--max_examples', default=3, type=int)
    parser.add_argument('--fewshot_override', default=None, type=int)


    args = parser.parse_args()

    if args.model == 'debug':
        args.model = DEBUG_MODEL

    lm_obj = HFLM(
        pretrained=args.model,
        device='cuda',
        batch_size=1,
    )

    lm_eval.tasks.initialize_tasks()

    results = lm_eval.simple_evaluate(
        model=lm_obj,
        limit=args.max_examples,
        num_fewshot=args.fewshot_override,
        tasks=['pubmedqa_medprompt'],
    )

    print(results)
