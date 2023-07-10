Megatron ([1](https://arxiv.org/pdf/1909.08053.pdf), [2](https://arxiv.org/pdf/2104.04473.pdf), and [3](https://arxiv.org/pdf/2205.05198)) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository is for ongoing research on training large transformer language models at scale. We developed efficient, model-parallel ([tensor](https://arxiv.org/pdf/1909.08053.pdf), [sequence](https://arxiv.org/pdf/2205.05198), and [pipeline](https://arxiv.org/pdf/2104.04473.pdf)), and multi-node pre-training of transformer based models such as [GPT](https://arxiv.org/abs/2005.14165), [BERT](https://arxiv.org/pdf/1810.04805.pdf), and [T5](https://arxiv.org/abs/1910.10683) using mixed precision.

Below are some of the projects where we have directly used Megatron:
* [BERT and GPT Studies Using Megatron](https://arxiv.org/pdf/1909.08053.pdf)
* [BioMegatron: Larger Biomedical Domain Language Model](https://www.aclweb.org/anthology/2020.emnlp-main.379.pdf)
* [End-to-End Training of Neural Retrievers for Open-Domain Question Answering](https://arxiv.org/abs/2101.00408)
* [Large Scale Multi-Actor Generative Dialog Modeling](https://www.aclweb.org/anthology/2020.acl-main.8.pdf)
* [Local Knowledge Powered Conversational Agents](https://arxiv.org/abs/2010.10150)
* [MEGATRON-CNTRL: Controllable Story Generation with External Knowledge Using Large-Scale Language Models](https://www.aclweb.org/anthology/2020.emnlp-main.226.pdf)
* [RACE Reading Comprehension Dataset Leaderboard](http://www.qizhexie.com/data/RACE_leaderboard.html)
* [Training Question Answering Models From Synthetic Data](https://www.aclweb.org/anthology/2020.emnlp-main.468.pdf)
* [Few-shot Instruction Prompts for Pretrained Language Models to Detect Social Biases](https://arxiv.org/abs/2112.07868)
* [Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models](https://arxiv.org/abs/2202.04173)
* [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](https://arxiv.org/abs/2201.11990)
* [Multi-Stage Prompting for Knowledgeable Dialogue Generation](https://arxiv.org/abs/2203.08745)

Megatron is also used in [NeMo Megatron](https://developer.nvidia.com/nvidia-nemo#nemo-megatron), a framework to help enterprises overcome the challenges of building and training sophisticated natural language processing models with billions and trillions of parameters.

Our codebase is capable of efficiently training very large (hundreds of billions of parameters) language models with both model and data parallelism. To demonstrate how the code scales with multiple GPUs and model sizes, we consider GPT models from 1 billion all the way to 1 trillion parameters. All models use a vocabulary size of 51,200 and a sequence length of 2048. We vary hidden size, number of attention heads, and number of layers to arrive at a specifc model size. As the model size increases, we also modestly increase the batch size. We leverage [NVIDIA's Selene supercomputer](https://www.top500.org/system/179842/) to perform scaling studies and use up to 3072 [A100](https://www.nvidia.com/en-us/data-center/a100/) GPUs for the largest model. Each cluster node has 8 NVIDIA 80GB A100 GPUs. The graph below shows that we scale nearly linear up to 1 trillion parameter models running on 3072 GPUs. Note that these results are from benchmark runs and these models were not trained to convergence; however, the FLOPs are measured for end-to-end training, i.e., includes all operations including data loading, optimization, and even logging.

![Scaling Graph](images/Achieved_petaFLOPs.png)

The following table shows both model (MFU) and hardware (HFU) FLOPs utilization for select configurations up to 1T parameters (see [our paper](https://arxiv.org/pdf/2205.05198) for a description of how these are calculated). As the model size increases, we achieve better GPU utilization and for the one trillion parameter model, we reach a MFU and HFU of 56.3% and 57.0%, respectively. Note that these numbers are also measured on benchmark runs and in this case are measured using a data parallel size of one. Data parallelism introduces some overhead due to the gradient all-reduce required between the data parallel groups. However, for large transformer models, this overhead is not large and can almost entirely eliminted by overlapping the gradient all-reduce with backpropagation.

| Model Size | Model FLOPs Utilization | Hardware FLOPs Utilization |
| :---: | :---: | :---: |
| 22B   | 41.5% | 43.7% |
| 175B  | 51.4% | 52.8% |
| 530B  | 56.0% | 57.0% |
| 1T    | 56.3% | 57.0% |

# Contents
   * [Contents](#contents)
   * [Setup](#setup)
      * [Downloading Checkpoints](#downloading-checkpoints)
   * [Usage](#usage)
   * [Training](#training)
      * [Data Preprocessing](#data-preprocessing)
      * [BERT Pretraining](#bert-pretraining)
      * [GPT Pretraining](#gpt-pretraining)
      * [T5 Pretraining](#t5-pretraining)
      * [Distributed Pretraining](#distributed-pretraining)
      * [Activation Checkpointing and Recomputation](#activation-checkpointing-and-recomputation)
      * [Distributed Optimizer](#distributed-optimizer)
      * [GPT-3 Example](#gpt-3-example)
   * [Evaluation and Tasks](#evaluation-and-tasks)
      * [GPT Text Generation](#gpt-text-generation)
      * [GPT Evaluation](#gpt-evaluation)
         * [WikiText Perplexity Evaluation](#wikitext-perplexity-evaluation)
         * [LAMBADA Cloze Accuracy](#lambada-cloze-accuracy)
      * [BERT Task Evaluation](#bert-task-evaluation)
         * [RACE Evaluation](#race-evaluation)
         * [MNLI Evaluation](#mnli-evaluation)
   * [Datasets](#datasets)
      * [Collecting Wikipedia Training Data](#collecting-wikipedia-training-data)
      * [Collecting GPT Webtext Data](#collecting-gpt-webtext-data)

# Setup
We strongly recommend using the latest release of [NGC's PyTorch container](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch). If you can't use this for some reason, use the latest pytorch, cuda, nccl, and NVIDIA [APEX](https://github.com/NVIDIA/apex#quick-start) releases.  Data preprocessing requires [NLTK](https://www.nltk.org/install.html), though this is not required for training, evaluation, or downstream tasks.

## Downloading Checkpoints
We have provided pretrained [BERT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_bert_345m) and [GPT-345M](https://ngc.nvidia.com/catalog/models/nvidia:megatron_lm_345m) checkpoints for use to evaluate or finetuning downstream tasks. To access these checkpoints, first [sign up](https://ngc.nvidia.com/signup) for and [setup](https://ngc.nvidia.com/setup/installers/cli) the NVIDIA GPU Cloud (NGC) Registry CLI. Further documentation for downloading models can be found in the [NGC documentation](https://docs.nvidia.com/dgx/ngc-registry-cli-user-guide/index.html#topic_6_4_1).

Alternatively, you can directly download the checkpoints using:

<pre>
BERT-345M-uncased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_uncased/zip -O megatron_bert_345m_v0.1_uncased.zip
BERT-345M-cased: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_bert_345m/versions/v0.1_cased/zip -O megatron_bert_345m_v0.1_cased.zip
GPT-345M: wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
</pre>

The models require vocabulary files to run. The BERT  WordPiece vocab file can be extracted from Google's pretrained BERT models: [uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt), [cased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt). The GPT [vocab file](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json) and [merge table](https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt) can be downloaded directly.

# Usage

After installation, there are several possible workflows. The most comprehensive is:
1. Data preprocessing
2. Pretraining
3. Finetuning (Optional for zero-shot tasks)
4. Downstream task evaluation or text generation

However, steps 1 and 2 can be replaced by using one of the pretrained models mentioned above.

We've provided several scripts for pretraining both BERT and GPT in [`examples`](./examples) directory, as well as scripts for both zero-shot and fine-tuned downstream tasks including MNLI, RACE, WikiText103, and LAMBADA evaluation. There is also a script for GPT interactive text generation.

# Training
## Data Preprocessing
The training data requires preprocessing. First, place your training data in a loose json format, with one json containing a text sample per line. For example:
<pre>
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
</pre>

The name of the `text` field of the json can be changed by using the `--json-key` flag in [`preprocess_data.py`](./tools/preprocess_data.py) The other metadata are optional and are not used in training.

The loose json is then processed into a binary format for training. To convert the json into mmap, cached index file, or the lazy loader format use `preprocess_data.py`. Set the `--dataset_impl` flag to `mmap`, `cached`, or `lazy`, respectively (default is `mmap`). An example script to prepare data for BERT training is:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output_prefix my-bert \
       --vocab bert-vocab.txt \
       --dataset_impl mmap \
       --tokenizer_type BertWordPieceLowerCase \
       --split_sentences
</pre>

The output will be two files named, in this case, `my-bert_text_sentence.bin` and `my-bert_text_sentence.idx`. The `--data_path` specified in later BERT training is the full path and new filename, but without the file extension.

For T5 use the same preprocessing as BERT, perhaps renaming it to:
<pre>
       --output_prefix my-t5 \
</pre>

Some minor modifications are required for GPT data preprocessing, namely, the addition of a merge table, an end-of-document token, removal of sentence splitting, and a change to the tokenizer type:
<pre>
python tools/preprocess_data.py \
       --input my-corpus.json \
       --output_prefix my-gpt2 \
       --vocab gpt2-vocab.json \
       --dataset_impl mmap \
       --tokenizer_type GPT2BPETokenizer \
       --merge_file gpt2-merges.txt \
       --append_eod
</pre>

Here the output files are named `my-gpt2_text_document.bin` and `my-gpt2_text_document.idx`. As before, in GPT training, use the longer name without the extension as `--data_path`.

Further command line arguments are described in the source file [`preprocess_data.py`](./tools/preprocess_data.py).

## BERT Pretraining


The `examples/pretrain_bert.sh` script runs single GPU 345M parameter BERT pretraining. Debugging is the primary use for single GPU training, as the code base and command line arguments are optimized for highly distributed training. Most of the arguments are fairly self-explanatory. By default, the learning rate decays linearly over the training iterations starting at `--lr` to a minimum set by `--min_lr` over `--lr_decay_iters` iterations. The fraction of training iterations used for warmup is set by `--lr_warmup_fraction`. While this is single GPU training, the batch size specified by `--micro_batch_size` is a single forward-backward path batch-size and the code will perform gradient accumulation steps until it reaches `global_batch_size` which is the batch size per iteration. The data is partitioned into a 949:50:1 ratio for training/validation/test sets (default is 969:30:1). This partitioning happens on the fly, but is consistent across runs with the same random seed (1234 by default, or specified manually with `--seed`). We use `train_iters` as the training iterations requested. Alternatively, one can provide `--train_samples` which is total number of samples to train on. If this option is present, then instead of providing `--lr_decay_iters`, one will need to provide `--lr_decay_samples`.

The logging, checkpoint-saving, and evaluation intervals are specified. Checkpointing the activations facilitates the training of larger models and/or batches. Note that the `--data_path` now includes the additional `_text_sentence` suffix added in preprocessing, but does not include the file extensions.

<pre>
CHECKPOINT_PATH=checkpoints/bert_345m
VOCAB_FILE=bert-vocab.txt
DATA_PATH=my-bert_text_sentence

BERT_ARGS="--num_layers 24 \
           --hidden_size 1024 \
           --num_attention_heads 16 \
           --seq_length 512 \
           --max_position_embeddings 512 \
           --lr 0.0001 \
           --lr_decay_iters 990000 \
           --train_iters 2000000 \
           --min_lr 0.00001 \
           --lr_warmup_fraction 0.01 \
	       --micro_batch_size 4 \
           --global_batch_size 8 \
           --vocab_file $VOCAB_FILE \
           --split 949,50,1 \
           --fp16"

OUTPUT_ARGS="--log_interval 10 \
             --save_interval 500 \
             --eval_interval 100 \
             --eval_iters 10 \
             --activations_checkpoint_method uniform"

python pretrain_bert.py \
       $BERT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH
</pre>

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).

## GPT Pretraining

The `examples/pretrain_gpt.sh` script runs single GPU 345M parameter GPT pretraining. As mentioned above, single GPU training is primarily intended for debugging purposes, as the code is optimized for distributed training.

It follows largely the same format as the previous BERT script with a few notable differences: the tokenization scheme used is BPE (which requires a merge table and a `json` vocabulary file) instead of WordPiece, the model architecture allows for longer sequences (note that the max position embedding must be greater than or equal to the maximum sequence length), and the `--lr_decay_style` has been set to cosine decay.  Note that the `--data_path` now includes the additional `_text_document` suffix added in preprocessing, but does not include the file extensions.

<pre>
CHECKPOINT_PATH=checkpoints/gpt2_345m
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
DATA_PATH=my-gpt2_text_document

GPT_ARGS="--num_layers 24 \
          --hidden_size 1024 \
          --num_attention_heads 16 \
          --seq_length 1024 \
          --max_position_embeddings 1024 \
          --micro_batch_size 4 \
          --global_batch_size 8 \
          --lr 0.00015 \
          --train_iters 500000 \
          --lr_decay_iters 320000 \
          --lr_decay_style cosine \
          --vocab_file $VOCAB_FILE \
          --merge_file $MERGE_FILE \
          --lr_warmup_fraction .01 \
          --fp16"

OUTPUT_ARGS=&#60;same as those in <a href="#bert-pretraining">BERT pretraining</a> above&#62;

python pretrain_gpt.py \
       $GPT_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH \
</pre>

Further command line arguments are described in the source file [`arguments.py`](./megatron/arguments.py).

## T5 Pretraining

Very similar to BERT and GPT, the `examples/pretrain_t5.sh` script runs single GPU "base" (~220M parameter) T5 pretraining. The primary difference from BERT and GPT is the addition of the following arguments to accommodate the T5 architecture:

* `--kv_channels` sets the inner dimension of the "key" and "value" matrices of all attention mechanisms in the model. For BERT and GPT this defaults to the hidden size divided by the number of attention heads, but can be configured for T5.

* `--ffn_hidden_size` sets the hidden size in the feed-forward networks within a transformer layer. For BERT and GPT this defaults to 4 times the transformer hidden size, but can be configured for T5.

* `--encoder_seq_length` and `--decoder_seq_length` set the sequence length for the encoder and decoder separately.

All of the other arguments remain as they were for BERT and GPT pretraining.

<pre>
CHECKPOINT_PATH=checkpoints/t5_base
VOCAB_FILE=t5-vocab.txt
DATA_PATH=my-t5_text_sentence

T5_ARGS="--num_layers 24 \
         --hidden_size 1024 \
         --num_attention_heads 16 \
         --kv_channels 64 \
         --ffn_hidden_size 3072 \
         --encoder_seq_length 512 \
         --decoder_seq_length 128 \
         --max_position_embeddings 512 \
         --lr 0.0001 \
         --lr_decay_iters 990000 \
         --train_iters 2000000 \
         --min_lr 0.00001 \
         --lr_warmup_fraction 0.01 \
         --micro_batch_size 16 \
         --global_batch_size 2048 \
         --vocab_file $VOCAB_FILE \
         --vocab_extra_ids 100 \
         --split 949,50,1 \
         --fp16"

OUTPUT_ARGS=&#60;same as those in <a href="#bert-pretraining">BERT pretraining</a> above&#62;

python pretrain_t5.py \
       $T5_ARGS \
       $OUTPUT_ARGS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data_path $DATA_PATH
</pre>


## Distributed Pretraining

The `examples/pretrain_{bert,gpt,t5}_distributed.sh` scripts use the PyTorch distributed launcher for distributed training. As such, multi-node training can be achieved by properly setting environment variables and using `init_method='env://'` in the launcher. See the official PyTorch [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default, multi-node training uses the [nccl](https://developer.nvidia.com/nccl) distributed backend. A simple set of additional arguments and the use of the PyTorch distributed module with the Python flag `-m torch.distributed.launch`, detailed below, are the only additional requirements to adopt distributed training.

We use two types of parallelism: data and model parallelism. We facilitate two distributed data parallel implementations: a simple one of our own that performs gradient all-reduce at the end of back propagation step, and Torch's distributed data parallel wrapper that overlaps gradient reduction with back propagation computation. To switch between these two options use `--DDP_impl local` or `--DDP_impl torch`, respectively. As expected, Torch distributed data parallelism is more efficient at larger model sizes. For example, for the 8.3 billion parameters model running on 512 GPUs, the scaling increases from 60% to 76% when Torch's distributed data parallel is used. However, the overlapping method requires more memory and for some configurations (e.g., 2.5 billion parameters using 2-way model parallel and 1.2 billion parameters with no model parallel) can make the overall training slower as a result. We empirically found that using a smaller model in those cases improves the training time.

Second, we developed a simple and efficient two-dimensional model-parallel approach. To use tensor model parallelism (splitting execution of a single transformer module over multiple GPUs), add the `--tensor_model_parallel_size` flag to specify the number of GPUs among which to split the model, along with the arguments passed to the distributed launcher as mentioned above. To use sequence parallelism specify `--sequence_parallel`, which requires tensor model parallel as it split among the same GPUs.

To use pipeline model parallelism (sharding the transformer modules into stages with an equal number of transformer modules on each stage, and then pipelining execution by breaking the batch into smaller microbatches), use the `--pipeline_model_parallel_size` flag to specify the number of stages to split the model into (e.g., splitting a model with 24 transformer layers across 4 stages would mean each stage gets 6 transformer layers each).

<!-- The number of microbatches in a per-pipeline minibatch is controlled by the `--num_microbatches_in_minibatch` argument. With `WORLD_SIZE` GPUs, `TENSOR_MP_SIZE` tensor_model_parallel size, `PIPELINE_MP_SIZE` pipeline_model_parallel_size, `WORLD_SIZE`/(`TENSOR_MP_SIZE` * `PIPELINE_MP_SIZE`) GPUs will be used for data parallelism. The default values for `--tensor-model-parallel-size` and `--pipeline-model-parallel-size` is 1, which will not implement either form of model parallelism. -->

We have examples of how to use these two different forms of model parallelism the example scripts ending in `distributed_with_mp.sh`:

Other than these minor changes, the distributed training is identical to the training on a single GPU.

Distributed training:
<pre>
WORLD_SIZE=8
TENSOR_MP_SIZE=2
PIPELINE_MP_SIZE=2

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=&#60;same as above&#62;
VOCAB_FILE=&#60;same as above&#62;
DATA_PATH=&#60;same as above&#62;
MODEL_ARGS=&#60;same as above&#62;
OUTPUT_ARGS=&#60;same as above&#62;

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pretrain_<model>.py \
                $MODEL_ARGS \
                $OUTPUT_ARGS \
                --save $CHECKPOINT_PATH \
                --load $CHECKPOINT_PATH \
                --data_path $DATA_PATH \
                --tensor_model_parallel_size $TENSOR_MP_SIZE \
                --pipeline_model_parallel_size $PIPELINE_MP_SIZE \
                --sequence_parallel \
                --DDP_impl torch
</pre>

The interleaved pipelining schedule (more details in Section 2.2.2 of [our paper](https://arxiv.org/pdf/2104.04473.pdf)) can be enabled using the `--num_layers-per-virtual-pipeline-stage` argument, which controls the number of transformer layers in a virtual stage (by default with the non-interleaved schedule, each GPU will execute a single virtual stage with `NUM_LAYERS / PIPELINE_MP_SIZE` transformer layers). The total number of layers in the transformer model should be divisible by this argument value. Additionally, the number of microbatches in the pipeline (computed as `GLOBAL_BATCH_SIZE / (DATA_PARALLEL_SIZE * MICRO_BATCH_SIZE)`) should be divisible by the `PIPELINE_MP_SIZE` when using this schedule (this condition is checked in an assertion in the code). The interleaved schedule is not supported for pipelines with 2 stages (`PIPELINE_MP_SIZE=2`).

## Activation Checkpointing and Recomputation

To reduce GPU memory usage so deploy a large model to a training system, we support activation checkpointing and recomputation. We support two levels of recompute granularity: `selective` and `full`. Selective recomputation is the default and recommended in almost all cases. It saves the activations that take less space and are expensive to recompute and recomputes activations that take a lot of space but are relatively cheap to recompute (see [our paper](https://arxiv.org/pdf/2205.05198) for details). To enable selective activation recompute simply use `--recompute_activations`.

For cases where memory is very tight, `full` checkpointing saves just the inputs to a transformer layer, or a block of transformer layers, and recomputes everything else. To turn on full activation recompute use `--recompute_granularity full`. When using full activation recomputation, there are two methods: `uniform` and `block`, chosen using the `--recompute_method` argument.

* Uniform method uniformly divides the Transformer layers into groups of layers and stores the input activations of each group in the memory. The baseline group size is 1 and, in this case, the input activation of each Transformer layer is checkpointed. When the GPU memory is insufficient, increasing the number of layers per group reduces the memory usage thus enables running a bigger model. For example, when using the number of layers per group of 4, the input activation of each group of 4 Transformer layers is checkpointed.

* Block method checkpoints the input activations of a set number of individual Transformer layers per pipeline stage and do the rest of layers without any checkpointing. This method can be used to skip checkpointing some Transformer layers until the GPU memory is fully used, which is applicable only when there is unused GPU memory. Checkpointing fewer transformer layers avoids unnecessary activation recomputation in the backprop thus improves training performance. For example, when we specify 5 layers to checkpoint of 8 layers per pipeline stage, the input activations of only the first 5 Transformer layers are checkpointed and activation recomputation for the rest 3 layers is not needed in the backprop.


## Distributed Optimizer

Usage: `--use_distributed_optimizer`. Compatible with all model and data types.

The distributed optimizer is a memory savings technique, whereby the optimizer state is evenly distributed across data parallel ranks (versus the traditional method of replicating the optimizer state across data parallel ranks). As described in [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), our implementation distributes all optimizer state that does not overlap with the model state. For example, when using fp16 model params, the distributed optimizer maintains its own separate copy of fp32 main params & grads, which are distributed across DP ranks. When using bf16 model params, however, the distributed optimizer's fp32 main grads are the same as the model's fp32 grads, and so the grads in this case are not distributed (although the fp32 main params are still distributed, as they are separate from the bf16 model params).

Theoretical memory savings vary depending on the combination of the model's param dtype and grad dtype. In our implementation, the theoretical number of bytes per parameter is (where 'd' is the data parallel size):

| | Non-distributed optim | Distributed optim |
|-|-|-|
| fp16 param, fp16 grads | 20 | 4 + 16/d |
| bf16 param, fp32 grads | 18 | 6 + 12/d |
| fp32 param, fp32 grads | 16 | 8 + 8/d |

## FlashAttention

Usage: `--use-flash-attn`. Support attention head dimensions at most 128.

[FlashAttention](https://github.com/HazyResearch/flash-attention) is a fast and
memory-efficient algorithm to compute exact attention. It speeds up model
training and reduces memory requirement.

To install FlashAttention:
```sh
pip install flash-attn
```

## GPT-3 Example

In `examples/pretrain_gpt3_175B.sh` we have provided an example of how to configure Megatron to run [GPT-3](https://arxiv.org/abs/2005.14165) with 175 billion parameters on 1024 GPUs. The script is designed for [slurm](https://slurm.schedmd.com/documentation.html) with [pyxis](https://github.com/NVIDIA/pyxis) plugin but can be easily adopted to any other scheduler. It uses 8-way and 16-way tensor and pipeline parallelism, respectively. With options `global_batch_size 1536` and `rampup_batch_size 16 16 5859375`, the training will start with global batch size 16 and linearly increase the global batch size to 1536 over 5,859,375 samples with incrmeental steps 16. The training dataset can be either a single set or a multiple datasets combined with a set of weights.

With full global batch size of 1536 on 1024 A100 GPUs, each iteration takes around 32 seconds resulting in 138 teraFLOPs per GPU which is 44% of the theoretical peak FLOPs.

# Evaluation and Tasks

We provide several command line arguments, detailed in the scripts listed below, to handle various zero-shot and fine-tuned downstream tasks. However, you can also finetune your model from a pretrained checkpoint on other corpora as desired. To do so, simply add the `--finetune` flag and adjust the input files and training parameters within the original training script. The iteration count will be reset to zero, and the optimizer and internal state will be reinitialized. If the fine-tuning is interrupted for any reason, be sure to remove the `--finetune` flag before continuing, otherwise the training will start again from the beginning.

Because evaluation requires substantially less memory than training, it may be advantageous to merge a model trained in parallel for use on fewer GPUs in downstream tasks. The following script accomplishes this. This example reads in a GPT model with 4-way tensor and 4-way pipeline model parallelism and writes out a model with 2-way tensor and 2-way pipeline model parallelism.

<pre>
python tools/checkpoint_util.py \
        --model_type GPT \
        --load_dir checkpoints/gpt3_tp4_pp4 \
        --save_dir checkpoints/gpt3_tp2_pp2 \
        --target_tensor_parallel_size 2 \
        --target-pipeline-paralle-size 2

</pre>

Several downstream tasks are described for both GPT and BERT models below. They can be run in distributed and model parallel modes with the same changes used in the training scripts.

## GPT Text Generation

We have included a simple REST server to use for text generation in `tools/run_text_generation_server.py`. You run it much like you would start a pretraining job, specifying an appropriate pretrained checkpoint. There are also few optional parameters: `temperature`, `top-k`and `top-p`. See `--help` or the source file for more information. See [examples/run_text_generation_server_345M.sh](examples/run_text_generation_server_345M.sh) for an example of how to run the server.

Once the server is running you can use `tools/text_generation_cli.py` to query it, it takes one argument which is the host the server is running on.

<pre>
tools/text_generation_cli.py localhost
</pre>

You can also use CURL or any other tools to query the server directly:

<pre>
curl 'http://localhost:5000/api' -X 'PUT' -H 'Content-Type: application/json; charset=UTF-8'  -d '{"prompts":["Hello world"], "tokens_to_generate":1}'
</pre>

See [megatron/text_generation_server.py](megatron/text_generation_server.py) for more API options.

## GPT Evaluation
We include example scripts for GPT evaluation on WikiText perplexity evaluation and LAMBADA Cloze accuracy.

### WikiText Perplexity Evaluation
For even comparison with prior works, we evaluate perplexity on the word-level [WikiText-103 test dataset](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.

We use the following command to run WikiText-103 evaluation on a 345M parameter model.
<pre>
TASK="WIKITEXT103"

VALID_DATA=&#60;wikitext path&#62;.txt
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m

COMMON_TASK_ARGS="--num_layers 24 \
                  --hidden_size 1024 \
                  --num_attention_heads 16 \
                  --seq_length 1024 \
                  --max_position_embeddings 1024 \
                  --fp16 \
                  --vocab_file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid_data $VALID_DATA \
       --tokenizer_type GPT2BPETokenizer \
       --merge_file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro_batch_size 8 \
       --activations_checkpoint_method uniform \
       --log_interval 10 \
       --no_load_optim \
       --no_load_rng
</pre>


### LAMBADA Cloze Accuracy
To compute LAMBADA cloze accuracy (the accuracy of predicting the last token given the preceding tokens) we utilize a detokenized, processed version of the [LAMBADA dataset](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).

We use the following command to run LAMBADA evaluation on a 345M parameter model. Note that the `--strict_lambada` flag should be used to require whole word matching. Make that `lambada` is part of the file path.

<pre>
TASK="LAMBADA"

VALID_DATA=&#60;lambada path&#62;.json
VOCAB_FILE=gpt2-vocab.json
MERGE_FILE=gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m
COMMON_TASK_ARGS=&#60;same as those in <a href="#wikitext-perplexity-evaluation">WikiText Perplexity Evaluation</a> above&#62;

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid_data $VALID_DATA \
       --tokenizer_type GPT2BPETokenizer \
       --strict_lambada \
       --merge_file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro_batch_size 8 \
       --activations_checkpoint_method uniform \
       --log_interval 10 \
       --no_load_optim \
       --no_load_rng
</pre>

Further command line arguments are described in the source file [`main.py`](./tasks/main.py)

## BERT Task Evaluation
### RACE Evaluation
The following script finetunes the BERT model for evaluation on the [RACE dataset](http://www.cs.cmu.edu/~glai1/data/race/). The `TRAIN_DATA` and `VALID_DATA` directory contain the RACE dataset as separate `.txt` files. Note that for RACE, the batch size is the number of RACE query's to evaluate. Since each RACE query has four samples, the effective batch size passed through the model will be four times the batch size specified on the command line.

<pre>
TRAIN_DATA="data/RACE/train/middle"
VALID_DATA="data/RACE/dev/middle \
            data/RACE/dev/high"
VOCAB_FILE=bert-vocab.txt
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
CHECKPOINT_PATH=checkpoints/bert_345m_race
COMMON_TASK_ARGS="--num_layers 24 \
                  --hidden_size 1024 \
                  --num_attention_heads 16 \
                  --seq_length 512 \
                  --max_position_embeddings 512 \
                  --fp16 \
                  --vocab_file $VOCAB_FILE"

COMMON_TASK_ARGS_EXT="--train_data $TRAIN_DATA \
                      --valid_data $VALID_DATA \
                      --pretrained_checkpoint $PRETRAINED_CHECKPOINT \
                      --activations_checkpoint_method uniform \
                      --save_interval 10000 \
                      --save $CHECKPOINT_PATH \
                      --log_interval 100 \
                      --eval_interval 1000 \
                      --eval_iters 10 \
                      --weight_decay 1.0e-1"

python tasks/main.py \
       --task RACE \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer_type BertWordPieceLowerCase \
       --epochs 3 \
       --micro_batch_size 4 \
       --lr 1.0e-5 \
       --lr_warmup_fraction 0.06
</pre>

### MNLI Evaluation
The following script finetunes the BERT model for evaluation with the [MultiNLI sentence pair corpus](https://www.nyu.edu/projects/bowman/multinli/). Because the matching tasks are quite similar, the script can be quickly tweaked to work with the [Quora Question Pairs](https://www.kaggle.com/quora/question-pairs-dataset) (QQP) dataset as well.

<pre>

TRAIN_DATA="data/glue_data/MNLI/train.tsv"
VALID_DATA="data/glue_data/MNLI/dev_matched.tsv \
            data/glue_data/MNLI/dev_mismatched.tsv"
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
VOCAB_FILE=bert-vocab.txt
CHECKPOINT_PATH=checkpoints/bert_345m_mnli
COMMON_TASK_ARGS=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;
COMMON_TASK_ARGS_EXT=&#60;same as those in <a href="#race-evaluation">RACE Evaluation</a> above&#62;

python tasks/main.py \
       --task MNLI \
       $COMMON_TASK_ARGS \
       $COMMON_TASK_ARGS_EXT \
       --tokenizer_type BertWordPieceLowerCase \
       --epochs 5 \
       --micro_batch_size 8 \
       --lr 5.0e-5 \
       --lr_warmup_fraction 0.065
</pre>

# Datasets
We do not host any datasets for GPT or BERT training, however, we detail their collection so that our results may be reproduced.

## Collecting Wikipedia Training Data
We recommend following the Wikipedia data extraction process specified by Google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text."

We recommend using the `--json` argument when using WikiExtractor, which will dump the Wikipedia data into loose json format (one json per line), making it more manageable on the file system and also readily consumable by our codebase. We recommend further preprocessing this json dataset by nltk punctuation standardization. For BERT training, use the `--split_sentences` flag to `preprocess_data.py` as described [above](#data-preprocessing) to include sentence breaks in the produced index. If you'd like to use Wikipedia data for GPT training you should still clean it with nltk/spacy/ftfy, but do not use the `--split-sentences` flag.

## Collecting GPT Webtext Data
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library from [jcpeterson](https://github.com/jcpeterson/openwebtext) and [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in our [openwebtext](./tools/openwebtext) directory. For reddit URLs corresponding to content up to October 2018 we arrived at approximately 37GB of content.
