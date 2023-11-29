# Frequently Asked Questions

## How to add special tokens?

When defining a new task, it is often needed to introduce tokens with special meanings.
For instance, let's say we want to add two tokens `[formula]` and `[/formula]` to indicate the start and end of a formula in mathematics textbooks.
In order to include these new tokens, you need to indicate them in three different places:

1. When tokenizing (`tools/preprocess_data.py`), use the flag `--vocab_extra_ids_list` with the new tokens:
    ```
    python tools/preprocess_data.py --vocab_extra_ids_list "[formula],[/formula]" # ...
    ```

1. When sharding the model (`tools/checkpoint_util.py`), using `--true_vocab_size`.
   For instance, Falcon has 65024 tokens by default.
   Including these two extra tokens will result in
   ```
   python tools/checkpoint_util.py --true_vocab_size 65026 # ...
   ```

1. When training (`finetune.py`) using `--vocab_extra_ids_list`.
   Same as before:
    ```
    python finetune.py --vocab_extra_ids_list "[formula],[/formula]" # ...
    ```

(tp-pp)=
## How to set TP and PP?

General strategies:
- It is recommended to use data parallelism as much as possible, only use model parallelism if the model cannot fit in the GPU or the micro batch size is very small.
- It is preferable to use tensor parallelism before pipeline parallelism, when working on a single machine.
- When a model does not fit in a single node, use a tensor parallelism level of as many GPUs each node has, and pipeline parallelism level as small as possible to allow the model to fit in memory, and maintain a micro batch size large enough (of at least 5).

In the codebase, you won't set data parallelism explicitly.
Rather, the data parallelism will be inferred automatically to be as high as possible, depending in your available hardware and TP, PP levels.
In general, the number of GPUs you need is:
```
GPUs = DP * TP * PP
```
For instance, if you have two nodes with 8 GPUs each, TP=4 and PP=2, then DP will be automatically set to 2 as `4 x 2 x 2 = 16`.

```{seealso}
- For more information on data and model parallelism see: https://huggingface.co/docs/transformers/v4.15.0/parallelism.
- Detailed information on how TP and PP works: https://arxiv.org/abs/2104.04473.
```

## How to launch training on multiple nodes?

In order to launch training on multiple nodes, you will set the appropriate arguments to the `torchrun` program.

1. Select a "master" or main node and take note of its IP address.
1. Launch the `finetune.py` script in the main node using `torchrun` with the following arguments:
   ```
   torchrun --n_proc_per_node NUMBER_OF_GPS_PER_NODE \
   	--nodes NUMBER_OF_NODES \
   	--node_rank 0 \
   	--master_addr ADDRESS_OF_THE_MAIN_NODE \
   	--master_port PORT \
   	finetune.py # ...
   ```
1. In the rest of the nodes, launch `finetune.py` with the same arguments, modifying `--node_rank` to a different value per node.

```{seealso}
- Take a look at the example script `examples/finetune.sh` for more information.
- Look at the [How to set TP and PP?](#tp-pp) section for more information.
```

## What are the basic hardware requirements?

A brief overview on the minimal training hardware requirements we observed during our experiments.

| Model      | min VRAM | tp  | pp  |
| :--------- | :------: | :-: | :-: |
| LLaMa2-7B  | 2x 80GB  | 2   | 1   |
| Mistral-7B | 4x 80GB  | 4   | 1   |
| Falcon-40B | 16x 80GB | 8   | 2   |
| LLaMa2-70B | 32x 80GB | 8   | 4   |

Note that you might observe different values depending on the sequence length, batch size and other configurations.

(shard)=
## How to shard and merge models?

Use `tools/checkpoint_util.py` to set the desired tensor and pipeline parallelism levels.

```
python tools/checkpoint_util.py \
	--target_tensor_parallel_size TP \
	--target_pipeline_parallel_size PP \
	--load_dir /path/to/original/weights/ \
	--save_dir /path/to/new/weights/ \
	--model_type MODEL \
	--bf16
```
Where MODEL can be either llama, llama2, falcon, gpt or bert, and TP and PP are the model parallelism levels desired.
Note that you can convert sharded weights (i.e. TP, PP > 1) to unsharded weights (TP = PP = 1) or viceversa.

## What arguments are used to train LLaMa 2?

We set the same hyperparamters specified by Meta during finetuning (see [their paper for more information](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)).
This means, that training LLaMa 2 7B can be done with the following arguments:

```bash
torchrun \
	# torchrun arguments # \
	--nproc_per_node <GPUs per node> \
	--nnodes <number of nodes> \
	--node_rank <0,1,2,etc a different number per node> \
	--master_addr <address of main node> \
	--master_port <port> \
	finetune.py --model_name llama2 \
	# hardware/distributed arguments # \
	--tensor_model_parallel_size <tp size> \
	--pipeline_model_parallel_size <pp>  \
	--bf16 \
	# training arguments # \
	--train_iters <train iters> \
	--adam_beta1 0.9 \
	--adam_beta2 0.95 \
	--adam_eps 1e-5 \
	--lr_decay_style cosine 5 \
	--lr_warmup_iters <warmup iters> \
	--lr 3e-4 \
	--min_lr 1e-6 \
	--weight_decay 0.1 \
	--micro_batch_size 5 \
	--global_batch_size 1000 \
	# additional optimization arguments # \
	--use_flash_attn \
	--sequence_parallel \
	--recompute_granularity selective \
	# logging/pathing arguments # \
	--load <path to megatron-llama> \
	--use_checkpoint_args \
	--vocab_file <path to tokenizer.model provided by Meta> \
	--log_interval 1 \
	--data_path <path to tokenized data> \
	--tokenizer_type SentencePieceTokenizer
```

```{seealso}
The file `examples/finetune.sh` gives the full picture of the arguments used to train either LLaMa.
```

## How to convert a LLaMa or Falcon architecture from a non-official checkpoint?

If you want to convert weights from a checkpoint other than the checkpoints provided by `llama-meta` or `tiiuae`, you might use `--model-path` during conversion.
For instance, to convert the [OpenAssistant llama2 70B](https://huggingface.co/OpenAssistant/llama2-70b-oasst-sft-v10) weights, run:

```
python weights_conversion/hf_to_megatron.py llama2 --size=70 \
	--out=/path/to/megatron/weights/ --cache-dir=/path/to/llama-2-7b/ \
	--model-path=OpenAssistant/llama2-70b-oasst-sft-v10
```

The `--model-path` argument should be either a local folder or the name of a model hosted on huggingface.

## I'm getting a `17300 Bus error (core dumped)` error!

If you are using a docker container and you get this error when sharding a large model, you might need to increase the shared memory size.
This is done via the command line option `--shm-size=128gb`.

## I'm getting a `ImportError: cannot import name 'helpers' from 'megatron.data'` error!

You need to compile the `helpers` module:

```
cd megatron/data
make
cd ../../
```
