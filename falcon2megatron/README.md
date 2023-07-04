# Falcon and Megatron

This file documents the usage of all currently-implemented falcon features in the megatron project.
The main features are:
1. **Falcon weight converting**.
   Converting existing falcon weights to a megatron-compatible format.
1. **Falcon correctness verification**.
   Verifying that the current implementation of falcon in the megatron code matches the official implementation.
1. **Falcon training**.
   Train falcon :D

## Falcon weight converting

**Warning**: The current code does not support model-parallelism, this is still work in progress.

The main file to do the weight conversion is `falcon2megatron/falcon2megatron.py`.
This file fetches the weights directly from the [huggingface implementation](https://huggingface.co/tiiuae/falcon-40b), so no additional files are required to run the script.
To extract, the 40B model for instance, run:
```
python falcon2megatron.py --size=40 --out=/path/to/output/directory/
```
This uses huggingface default cache directory to store the original weights.
To change the cache use:
```
python falcon2megatron.py --size=40 --out=/path/to/output/directory/ --cache-dir=/path/to/huggingface/cache/directory/
```

## Falcon correctness verification

To verify that the current implementation of falcon in the megatron code is correct, use the file `verify_falcon.py`.
See for instance `examples/verify_falcon.sh`.
Make sure to set the `--model_size=7 or `40` (depending on whether you test the 7B or 40B) and to use the /path/to/output/directory/ selected in the previous step as the `--load` argument.
Example outputs at this stage are:
```
...
Iteration 0...
Max absoulute error in the logits: 0.029 Abs loss error: 0.002
Iteration 1...
Max absoulute error in the logits: 0.037 Abs loss error: 0.002
Iteration 2...
Max absoulute error in the logits: 0.067 Abs loss error: 0.001
Iteration 3...
Max absoulute error in the logits: 0.051 Abs loss error: 0.001
...
```

Also, make sure to remove the `--bfp16` flag to use the 32-bit model.

## Falcon training

Use the `finetune_falcon.py`.
See for instance `examples/finetune_falcon.sh`.
