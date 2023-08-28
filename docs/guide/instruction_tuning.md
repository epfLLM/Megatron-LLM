# Instruction finetuning

This tutorial will guide you through the basics of instruction finetuning using the Megatron-LLM codebase, using LLaMa 2 as the base network.
See also the [getting started](getting_started) guide for information regarding installation of dependencies, pretraining, and weight preparation.
Following said tutorial, you would be able to finetune a 7B model in this guide, but feel free to use a different size.
In order to use Falcon, see the comments specified in the [getting started](getting_started) guide to learn more about the differences when using either model.

## Preparing raw data

The dataset used in this guide will be a subset of the [orca](https://huggingface.co/datasets/Open-Orca/OpenOrca) dataset, a general purpose instruction dataset.
We choose to only include the chain of thought instructions from the orca dataset in order to shrink the size of the data.
Feel free to use any other dataset, as long as the raw data is saved in `.jsonl` format, i.e. one `json` dictionary per line.
The dictionaries must include at least two keys (one for the "instruction" and another one for the expected "answer"), plus an optional "system" key.
In order to retrieve the CoT subset of the orca dataset, use the following code:

```python
import json

from datasets import load_dataset

# the `cache_dir` is optional
dataset = load_dataset("Open-Orca/OpenOrca", cache_dir="/path/to/cache", split="train")
with open("/path/to/raw/data.jsonl", "w+") as f:
    for document in tqdm(dataset):
        if document["id"].startswith("cot."):
            f.write(json.dumps(document) + "\n")
```

## Data preprocessing

In this step we will tokenize the raw data to binary files for optimized data loading during training.
Run:
```
python instruct/preprocess_instruct_data.py \
	--input=/path/to/raw/data.jsonl \
	--output_prefix=/path/to/tokenized/orca \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=/path/to/llama/tokenizer.model \
	--chunk_size=32 \
	--workers=32 \
	--vocab_extra_ids_list "<|im_start|>,<|im_end|>" \
	--question_key=question \
	--answer_key=response \
	--system_key=system_prompt  # Optional
```

## Training

At this point, you should come up with a Megatron checkpoint ready to be trained (i.e. sharded with the desired parallelism levels).
Take a look at the [getting started](getting_started) guide to look how to transform LLaMa 2 checkpoints in the huggingface format to Megatron, and shard the weights.

To start training, use the `finetune.py`.
Example usage:
```bash
LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 6500 --lr_decay_style cosine --lr_warmup_iters 650 --lr 2e-5 --min_lr 2e-6"
DISTRIBUTED_ARGS="--nproc_per_node NUMBER_OF_GPUS --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
torchrun $DISTRIBUTED_ARGS finetune.py \
	--tensor_model_parallel_size 4 \
	--pipeline_model_parallel_size 1 \
	--load /path/to/sharded/weights/ \
	--save /path/to/sharded/weights/ \
	--tensorboard_dir /path/to/sharded/weights/tensorboard/ \
	--data_path /path/to/tokenized/orca \
	--model_name llama2 \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file=/path/to/megatron/weights/tokenizer.model \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 8 \
	--global_batch_size 64 \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	--data_type instruction \
	--variable_seq_lengths \
	--vocab_extra_ids_list "<|im_start|>,<|im_end|>" \
	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS
```

The arguments given for pretraining and instruction finetuning are very similar, with the key differences being the batch sizes, learning rates, and the inclusion of `--data_type instruction`, `--variable_seq_lengths` and `--vocab_extra_ids_list`.
With the selected global batch size of 64, in 6500 iterations the trainer will perform approximately three epochs.
This will take approximately 3h hours to run on a 8x 80GB A100 device (DP=2, TP=4, PP=1).

```{note}
If your `--load` checkpoint corresponds to a checkpoint already trained with the Megatron-LLM codebase (and not a checkpoint gotten after directly converting from the huggingface format for instance), you might want to define a `--save` directory that points somewhere else, to avoid overwritting previous checkpoints.
You might also want to include the `--finetune` argument to ignore the previous optimizer and RNG states.
```

## Model Deployment

Once the finetuning is over, you can follow the [getting started](getting_started) guide steps to unshard your weights and convert them to huggingface, in order to do specific evaluations and deployment.
