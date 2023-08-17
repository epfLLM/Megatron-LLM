import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push checkpoints in HF transformers format to the Huggingface Hub.",
        epilog="Example usage: python push_to_hub.py /path/to/checkpoint --hf_repo_name your_org/model_name --dtype bf16 --auth_token hf_ba..."
    )
    parser.add_argument(
        "model_name",
        help="Path to checkpoint or model name",
        type=str,
    )
    parser.add_argument(
        "--dtype",
        help="auto (default), bf16, fp16 or fp32",
        type=str,
        default="auto",
    )
    parser.add_argument(
        "--hf_repo_name",
        help="HuggingFace repository name",
        type=str,
    )
    parser.add_argument(
        "--auth_token",
        help="User access token (HuggingFace) used for model upload",
        type=str,
    )
    parser.add_argument(
        "--output_folder",
        help="Output folder path (e.g. for dtype conversion)",
        type=str,
    )
    parser.add_argument(
        "--max_shard_size",
        help="Maximum size for a checkpoint before being sharded (default: 10GB)",
        type=str,
        default="10GB",
    )
    parser.add_argument(
        "--unsafe",
        help="Disable safetensor serialization",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--rope_scaling_type",
        help="Overwrite rope scaling type (linear, dynamic)",
        type=str,
        default="linear",
    )
    parser.add_argument(
        "--rope_scaling_factor",
        help="Overwrite rope scaling factor (float >1.0)",
        type=float,
    )
    parser.add_argument(
        "--trust_remote_code",
        help="Allow custom model code",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    safe_serialization = not args.unsafe

    if args.dtype in ("float16", "fp16"):
        torch_dtype = torch.float16
    elif args.dtype in ("float32", "fp32"):
        torch_dtype = torch.float32
    elif args.dtype in ("bfloat16", "bf16"):
        torch_dtype = torch.bfloat16
    elif args.dtype == "auto":
        torch_dtype = None
    else:
        print(f"Unsupported dtpye: {args.dtype}")
        sys.exit(1)

    if not args.hf_repo_name and not args.output_folder:
        print(
            "Please specify either `--hf_repo_name` to push to HF or `--output_folder` "
            "to export the model to a local folder."
        )
        sys.exit(1)

    print(f"Loading tokenizer '{args.model_name}' ...")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print(f"Tokenizer: {type(tokenizer).__name__} (vocab_size: {len(tokenizer):,})")

    print("Special tokens:")
    for token in tokenizer.all_special_tokens:
        id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token}: {id}")
    print()

    print(f"Loading model '{args.model_name}' ({args.dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Model: {type(model).__name__} (num_parameters={model.num_parameters():,})")

    print("Model architecture:")
    print(model)
    print()

    if args.rope_scaling_type is not None and args.rope_scaling_factor is not None:
        assert args.rope_scaling_type in ("linear", "dynamic")
        assert args.rope_scaling_factor >= 1.0
        rope_scaling = {
            "type": args.rope_scaling_type,
            "factor": args.rope_scaling_factor,
        }
        print(
            f"Setting new rope_scaling config: {rope_scaling} (old: {model.config.rope_scaling})"
        )
        model.config.rope_scaling = rope_scaling

    print("Model configuration:")
    print(model.config)
    print()

    if args.output_folder:
        print(f"Saving model to: {args.output_folder}")
        model.save_pretrained(
            args.output_folder,
            max_shard_size=args.max_shard_size,
            safe_serialization=safe_serialization,
        )

        print(f"Saving tokenizer to: {args.output_folder}")
        tokenizer.save_pretrained(args.output_folder)

    if args.hf_repo_name:
        print(f"Uploading model to HF repository ('{args.hf_repo_name}') ...")
        model.push_to_hub(
            args.hf_repo_name,
            use_auth_token=args.auth_token,
            max_shard_size=args.max_shard_size,
            safe_serialization=safe_serialization,
        )

        print(f"Uploading tokenizer to HF repository ('{args.hf_repo_name}') ...")
        tokenizer.push_to_hub(args.hf_repo_name, use_auth_token=args.auth_token)


if __name__ == "__main__":
    main()
