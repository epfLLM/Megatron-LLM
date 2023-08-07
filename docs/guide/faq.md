# Frequently Asked Questions

## How to add special tokens?

When defining a new task, it is often needed to introduce tokens with special meanings.
For instance, let's say we want to add two tokens `[formula]` and `[/formula]` to indicate the start and end of a formula in mathematics textbooks.
In order to include these new tokens, you need to indicate them in three different places:

1. When tokenizing (`tools/preprocess_data.py`), using the flag `--vocab_extra_ids_list` with the new tokens:
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

In this section we give a brief overview on the minimal hardware requirements we observed during our experiments.

| Model      | min VRAM |
| :--------- | :------: |
| LLaMa2-7B  | 2x 80GB  |
| LLaMa2-70B | 32x 80GB |
| Falcon-40B | ???      |


(shard)=
## How to shard and unshard models?

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
