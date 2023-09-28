### Normal

python main.py \
--model hf \
--model_args pretrained=EleutherAI/pythia-160m \
--tasks mimic-iii-rrs \
--device cuda:0 \
--limit 5

python scripts/write_out.py \
--tasks medmcqa \
--num_fewshot 2 \
--num_examples 5 \
--output_base_path write_out

### Multi-GPU Evaluation with Hugging Face accelerate

accelerate launch main.py \
--model hf \
--tasks mimic-iii-rrs \
--batch_size 16 

## Cite as

[x] mimic-iii-rrs<br/>
[x] mimic-cxr-rrs<br/>
[x] problem-list-sum<br/>
[x] consumer-health-questions<br/>
[x] dialogue2note-sum<br/>
[x] mednli<br/>
[x] medmcqa<br/>
[] pubmedqa (tanishq) <br/>
[] medqa_4options<br/>
