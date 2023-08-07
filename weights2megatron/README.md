# Welcome to Megatron

This file documents the usage of all currently-implemented falcon and llama features in the megatron project.
The main features are:
1. **Tokenizing**.
1. **Weight converting**.
   Converting existing weights to a megatron-compatible format.
1. **Correctness verification**.
   Verifying that the current implementation in the megatron code matches the official implementation.
1. **Checkpoint splitter**.
   Split the checkpoint saved in one file into multiple files, in order to use model parallelism.
1. **Training**.

## Tokenizing

More information available at `tokenize-utils/README.md`

## Weight converting

The main file to do the weight conversion is `weights2megatron/weights2megatron.py`.
When converting falcon weights, this file fetches the weights directly from the [huggingface implementation](https://huggingface.co/tiiuae/falcon-40b), so no additional files are required to run the script.
To extract, the 40B model for instance, run:
```
python weights2megatron.py falcon --size=40 --out=/path/to/output/directory/
```
This uses huggingface default cache directory to store the original weights.
To change the cache use:
```
python weights2megatron.py falcon --size=40 --out=/path/to/output/directory/ --cache-dir=/path/to/huggingface/cache/directory/
```

See also `examples/weights2megatron.sh`.

Llama weights are not so easily available, you need to [request them from meta](https://ai.meta.com/llama/).
In this case you also need to specify the directory specified as `--cache-dir` will be used to fetch the llama weights, for instance run:
```
python weights2megatron.py llama2 --size=7 --out=/path/to/output/directory/ --cache-dir=/path/to/meta/llama-2-7b/
```

**IMPORTANT**: If you are using megatron converted weights produced in the commit [332cf3c](https://github.com/epfLLM/Megatron-LLM/commit/332cf3cdb9b08a7dc26cb2764496378b58088012) or earlier, you will need to update your weights.
Use:
```
python weights2megatron/permute_qkv.py --input-dir=/path/to/old/checkpoint/ --output-dir=/path/to/new/checkpoint/
```

## Correctness verification


To verify that the current megatron code is correct, use the file `verify_correctness.py`.
See for instance `examples/verify.sh`.
Make sure to set the `--model_size=7` or `40` (depending on whether you test the 7B or 40B) and to use the /path/to/output/directory/ selected in the previous step as the `--load` argument.
Example outputs at this stage are:
```
Iteration 0...                                                                
Max absoulute error in the logits: 0.000143            
Abs loss error: 0.001175 Our loss: 2.021, theirs: 2.020
Iteration 1...                                                                
Max absoulute error in the logits: 0.000139
Abs loss error: 0.000886 Our loss: 1.813, theirs: 1.814
Iteration 2...
Max absoulute error in the logits: 0.000239
Abs loss error: 0.000741 Our loss: 1.808, theirs: 1.809
Iteration 3...
Max absoulute error in the logits: 0.000146
Abs loss error: 0.000657 Our loss: 1.756, theirs: 1.756
...
```

See also: `examples/verify.sh` script.

Also, make sure to remove the `--bfp16` flag to use the 32-bit model and get higher precision outputs.
However, if you use flash attention 2.0 running a 16 bit model is necessary as 32-float is not supported.

In order to verify llama make sure to convert the raw weights to huggingface.
See `convert_llama2hf.sh` and `convert_llama2hf.py` for more information.
Note that this step does not appy to llama2 as we use the official implementation to test it.
Also important for llama: make sure to add the `--no_new_tokens` flag during tokenizing (the `preprocess_data.py` script) and also during verification (the `verify_correctness.py` script).

**Note**: The current validation code does not support model-parallelism, to empirically test sharded models run `finetune.py` and verify that the loss makes sense.

## Checkpoint splitter

In order to use model parallelism you need to split the previously converted weights into multiple files.
To do this, use `tools/checkpoint_util.py`.
See for instance `examples/parallelize.sh`.
Once the weights are splitted, make sure to use the new checkpoint directory (set in the `--save_dir` argument) and the same tensor/model paralellism levels when running the next step.

**Note**: If you are using a docker image to run this step and get an error similar to `17300 Bus error (core dumped)` during this step, try increasing the shared memory size of the container (set to `--shm-size=128gb` in my experiments for falcon 40B).

## Training

Use the `finetune.py`.
See for instance `examples/finetune.sh`.
Don't forget to set the tensor and pipeline paralellism levels to the numbers set in the previous step.
The loss should decrease across iterations.

In order to use multi-node training using `examples/finetune.sh`, set the variables `--rank, --addr`.
For instance, to train a llama2-7b with `pp=1, dp=4, tp=4` on two nodes with 8xGPUs each, use:
```
# on node1
bash examples/finetune.sh llama2 --rank 0 --tp 4 --pp 1 --nodes 2 --addr host_address --size 7

# on node2
bash examples/finetune.sh llama2 --rank 1 --tp 4 --pp 1 --nodes 2 --addr host_address --size 7
```

Make sure to edit the paths in the script to match your local files.

**Important**: If you get an `ImportError: cannot import name 'helpers' from 'megatron.data'`, try running `cd megatron/data; make; cd ../../` to compile the `helpers` module.
