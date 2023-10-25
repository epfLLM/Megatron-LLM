# Getting started

This tutorial will guide you on the basic usage of Megatrom-LLM.
This guide we will fine tune a [LLaMa 2 7B](https://ai.meta.com/llama/) LLM on [code data](https://huggingface.co/datasets/bigcode/starcoderdata).
It is recommended to have at least 160GB VRAM available (e.g. two 80GB A100 GPUs).

```{note}
This tutorial can also be followed to train a Falcon architecture, using `falcon` instead of `llama2` throughout the guide.
```

## Setup

First we need to install the dependencies.


1. Clone our repo:
   ```
   git clone git@github.com:epfLLM/Megatron-LLM.git
   ```

1. Run the [nvcr docker image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch), mounting the source code to your desired path, e.g. `/mpt/Megatron-LLM`:
   ```
   sudo docker run --gpus all -it --rm \
   	-v /path/to/Megatron-LLM/:/mpt/Megatron-LLM \
   	nvcr.io/nvidia/pytorch:23.07-py3
   ```
   Note: "if you use Torch multiprocessing for multi-threaded data loaders, the default shared memory segment size that the container runs with may not be enough. Therefore, you should increase the shared memory size by issuing ... " (from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) `--shm-size=` as an argument to above command. E.g. `--shm-size=128gb`.

1. Enter the repository:
   ```
   cd /mpt/Megatron-LLM/
   ```

1. Install the additional dependencies not included in the `nvcr` image:
   ```
   pip install -r requirements.txt
   ```

1. Install the `megatron/data/helpers` binary:
   ```
   cd megatron/data/
   make
   cd ../../
   ```

(download_weights)=
## Downloading LLaMa2 weights

1. Request access to the weights directly to meta: https://ai.meta.com/resources/models-and-libraries/llama-downloads/.
1. Request access to the LLaMa2 huggingface model: https://huggingface.co/meta-llama/Llama-2-7b-hf.
1. Create a new huggingface token (or use an existing one): https://huggingface.co/settings/tokens.
1. Run the huggingface login CLI, and enter the token created on the previous step when asked:
   ```
   huggingface-cli login
   ```

## Preparing the raw data

:::{note}

This tutorial will use code data to fine tune the LLM.
Feel free to use any other dataset, as long as the raw data is saved in `.jsonl` format, i.e. one `json` dictionary with the key `"text"` per line:

```json
{"text": "The blue cat is big."}
{"text": "This is another document."}
```

In this case, skip to the [data preprocessing](#data-preprocessing) section.

:::

1. Accept starcoder's terms of use via the huggingface portal: https://huggingface.co/datasets/bigcode/starcoderdata
1. Create a huggingface token (or use an existing one) and login using `huggingface-cli` (see [Downloading LLaMa2 weights](#download_weights) for more information).
1. Download and save the starcoder dataset.
   In this tutorial we will use the `julia` data, but feel free to use any other subset.
   This data contains around 500M tokens.
   ```python
   import json
   from datasets import load_dataset

   # the `cache_dir` argument is optional
   dataset = load_dataset("bigcode/starcoderdata", data_dir="julia",
                          split="train", cache_dir="/path/to/cache/")
   with open("/path/to/raw.jsonl", "w+") as f:
       for document in dataset:
           document = {"id": document["id"], "text": document["content"]}
           f.write(json.dumps(document) + "\n")
   ```

At this point, the raw data will be available at `/path/to/raw.jsonl`.


(data-preprocessing)=
## Data preprocessing

In this step we will tokenize the raw data to binary files for optimized data loading during training.
Run:
```
python tools/preprocess_data.py --input=/path/to/raw.jsonl \
	--output_prefix=/path/to/tokenized/starcoder \
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=/path/to/tokenizer.model \
	--chunk_size=32 \
	--workers=16 \
	--no_new_tokens
```

```{note}
In this guide we use a sequence length of 1024 to accelerate training.
Note that the official sequence length of LLaMa2 is 4096.
```

```{note}
If you are using falcon, use `FalconTokenizer` instead of `SentencePieceTokenizer`, don't supply any `--vocab_file` and ignore the `--no_new_tokens` flag.
```


(weight-conversion)=
## Weight conversion

In order to use pretrained weights in the Megatron-LLM codebase, we will need to convert the official weights provided to be compatible with Megatron.
To do so, run:
```
python weights_conversion/hf_to_megatron.py llama2 --size=7 \
	--out=/path/to/megatron/weights/ --cache-dir=/path/to/llama-2-7b/
```

(correctness-verification)=
## Correctness verification (optional)

To make sure the weight conversion ran successfully we run the `verify_correctness.py` script.
This will run simultaneously the official LLaMa 2 implementation and the Megatron codebase.
Make sure to adjust the arguments to your convenience:
```bash
# arguments required by `torchrun`
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
LLAMA_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits --no_new_tokens --layernorm_epsilon 1e-5"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion"

torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name=llama2 \
	--model_size=7 \
	--load=/path/to/megatron/weights/ \
	--data_path=/path/to/tokenized/starcoder_text_document \  # without the .idx or .bin extension
	--tokenizer_type=SentencePieceTokenizer \
	--vocab_file=/path/to/megatron/weights/tokenizer.model \
	--huggingface_cache=/path/to/meta/llama-2-7b/ \
	--huggingface_device=cuda:1 \
	$COMMON_ARGS $LLAMA_ARGS  # dont include LLAMA_ARGS if using Falcon
```

This script will compare the logits output of Megatron model and the official implementation.
Expected outputs will yield average absolute error smaller than `0.01` when using 32-bit precision and `0.1` when using 16-bit precision.

## Model sharding

In order to use model parallelism you need to split the previously converted weights into multiple files, before you start training.
To do this, use `tools/checkpoint_util.py`.
Feel free to use different tensor parallel (tp) and pipeline (pp) sizes.
```
python tools/checkpoint_util.py \
	--target_tensor_parallel_size 2 \
	--target_pipeline_parallel_size 1 \
	--load_dir /path/to/megatron/weights/ \
	--save_dir /path/to/sharded/weights/ \
	--model_type llama2 \
	--true_vocab_size 32000 \
	--bf16
```

Feel free to set `--target_tensor_parallel_size` to 4 if you have 4 or more GPUs available.

## Training

Use the `finetune.py`.
Example usage:
```bash
LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 500 --lr_decay_style cosine --lr_warmup_iters 50 --lr 3e-4 --min_lr 1e-6"
DISTRIBUTED_ARGS="--nproc_per_node NUMBER_OF_GPUS --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"

torchrun $DISTRIBUTED_ARGS finetune.py \
	--tensor_model_parallel_size 2 \
	--pipeline_model_parallel_size 1 \
	--load /path/to/sharded/weights/ \
	--save /path/to/sharded/weights/ \
	--tensorboard_dir /path/to/sharded/weights/tensorboard/ \
	--data_path /path/to/tokenized/starcoder_text_document \  # without the .idx or .bin extension
	--model_name llama2 \
	--tokenizer_type SentencePieceTokenizer \
	--vocab_file=/path/to/megatron/weights/tokenizer.model \
	--bf16 \
	--use_flash_attn \
	--micro_batch_size 1 \
	--global_batch_size 1000 \
	--sequence_parallel \
	--recompute_granularity selective \
	--use_checkpoint_args \
	$COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $LLAMA_ARGS
```

With the selected global batch size of 1000, and the total number of training tokens around 500M, in 500 iterations the trainer will perform approximately one epoch.
Set your TP and PP values to the same numbers specified in the previous step.
This will take approximately 20 hours to run on an 8x 80GB A100 cluster (DP=2, TP=4, PP=1).
Feel free to increase the `--micro_batch_size` to speed up training.

:::{note}

To use distributed training make sure to set `nproc_per_node` to the number of GPUs per node, `nnodes` to the number of nodes in your training and `master_addr` to the address of your master node in the `DISTRIBUTED_ARGS` variable.
For instance, to train a two node cluster, with 8 GPUs each:
```
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
```

Then, run the `finetune.py` script in all your nodes with the same parameters, just setting a different `node_rank` at every node.

:::

```{seealso}
Take a look at `examples/finetune.sh for more information on the recommended hyperparameters
```

## Model Deployment

After training, merge your distributed weights again into a single model:
```
python tools/checkpoint_util.py \
	--target_tensor_parallel_size 1 \
	--target_pipeline_parallel_size 1 \
	--load_dir /path/to/sharded/weights/ \
	--save_dir /path/to/unsharded/trained/weights/ \
	--model_type llama2 \
	--true_vocab_size 32000 \
	--bf16
```

We provide a Megatron to Huggingface conversion utility for easier deployment: `weights_conversion/megatron_to_hf.py`.
Run:
```
python weights_conversion/megatron_to_hf.py --input_dir=/path/to/unsharded/trained/weights/ \
	--output_dir=/path/to/hf/weights/
```

Once the conversion is done, you can load the fine tuned weights using huggingface:
```python
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

pipeline = transformers.pipeline(
    "text-generation",
    model=LlamaForCausalLM.from_pretrained("/path/to/hf/weights/"),
    tokenizer=LlamaTokenizer.from_pretrained("/path/to/hf/weights/"),
    torch_dtype=torch.bfloat16,
    device="cuda"
)
prompt = """#= a function that returns the fibonacci number of its argument =#
function fibonacci(n::Int)::Int
"""
sequences = pipeline(prompt, max_new_tokens=100, do_sample=True, top_k=20,
                     num_return_sequences=1)
for sequence in sequences:
    print(sequence["generated_text"])
```

Once you are happy with your model performance, you might publish it to the huggingface hub using the `tools/push_to_hub.py` utility:

```
python tools/push_to_hub.py /path/to/hf/weights --hf_repo_name=MyRepoName/MyAwesomeModel --auth_token=MyAuthToken
```

## What's next?

1. Take a look at our example scripts to familiarize yourself with some other capabilities and hyperparameters used in the codebase, such as to train (pretrain or finetune) larger models:
   - `examples/parallelize.sh`
   - `examples/finetune.sh`
   - `examples/verify.sh`
1. See the [intruction finetuning](instruction_tuning) guide for more information on how to finetune a pretrained model to follow instructions.
1. Take a look at our [FAQ](faq) section.
1. See [Weights conversion](weights_conversion) for more information on the `hf_to_megatron.py` and `megatron_to_hf.py` scripts.
