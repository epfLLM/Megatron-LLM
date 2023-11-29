# Weights conversion

## Huggingface to megatron: `hf_to_megatron.py`

Convert weights from models in other formats (primarily huggingface) to megatron checkpoints.

This script supports converting Falcon, LLaMa and LLaMa 2 weights to megatron checkpoints.
Depending on the model to convert, the inputs might differ.

- **Falcon**/**Mistral**:
  Weights are automatically retrieved from the official implementation hosted in huggingface.
  Thus, the `--cache-dir` argument is optional, if specified it should point to
  the huggingface cache directory where the huggingface Falcon/Mistral weights will be stored.
  You will need to specify the `--size` argument to determine which version to download
  (i.e. Falcon 7B or 40B).
  Note that mistral only has 7B weights available.

- **LLaMa**, **LLaMa 2** and **CodeLlama**:
  Converting llama weights can be done either fetching the weights hosted
  in huggingface (recommended as it is the easier method) or directly from the
  weights provided by Meta.

  - From Meta weights (only available for LLaMa and LLaMa 2):
    You will need to specify the `--cache-dir` to the directory where the
    llama weights are stored.
    This will by default have the form `xB` (e.g. 7B or 70B) for llama v1,
    or `llama-2-xb` (e.g. llama-2-7b) for llama v2.

  - From huggingface weights:
    If `--cache-dir` is not specified or the directory specified does not
    contain the format expected from Meta weights, the converter will automatically
    retrieve the weights from huggingface, in which case the `--cache-dir` will
    have the same semantics as with Falcon.

    Note that to download llama v2 weights from huggingface, you will need to
    login using `huggingface-cli login` with a huggingface account which has been
    granted access to the `meta-llama/Llama-2-7b-hf` model.
        

In all cases, the megatron checkpoint will be stored in the `--out` argument.
If a huggingface is specified, the intermediate weights (i.e. the huggingface weights)
stored therein will not be removed when the conversion succeeds.

More information about the arguments:

```
positional arguments:
  {llama2,falcon,codellama,llama,mistral}

options:
  -h, --help            show this help message and exit
  --size {65,34,70,7,40,13,30}
                        The size of the model
  --out OUT             Directory to store the megatron weights (as checkpoint)
  --cache-dir CACHE_DIR
                        Directory to use as cache for the huggingface weights, or in case of the llama model, the path of the weights provided by Meta
```

## Megatron to huggingface: `megatron_to_hf.py`

Convert megatron checkpoints to huggingface weights.

This script will also convert the tokenizer configured.
Set the `--input_dir` to the megatron checkpoint root (i.e. where the
`latest_checkpointed_iteration.txt` file is located) and  `--output_dir` to
the directory where the huggingface weights should be stored.

More information about the arguments:

```
options:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Location of Megatron weights
  --num_output_shards NUM_OUTPUT_SHARDS
  --model {llama2,falcon,llama,codellama}
  --output_dir OUTPUT_DIR
                        Location to write HF model and tokenizer
  --cache_dir CACHE_DIR
                        Huggingface cache_dir (optional)
  --vocab_file VOCAB_FILE
                        Path to the vocab file
  --vocab_extra_ids_list VOCAB_EXTRA_IDS_LIST
                        comma separated list of special vocab ids to add to the tokenizer
  --override_special_tokens [OVERRIDE_SPECIAL_TOKENS ...]
                        One or more arguments to override special tokens. Syntax set as `key=value`, e.g. `eos=<|im_end|>`. Overrides available only bos,
                        cls, eos, mask, pad, sep, unk.
```
