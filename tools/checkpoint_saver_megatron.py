import os
import sys

import torch


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron saver')
    group.add_argument('--megatron_path', type=str, default=None,
                       help='Base directory of Megatron repository')
    group.add_argument('--target_tensor_parallel_size', type=int,
                       help='Target tensor model parallel size, defaults to the tensor parallel size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')
    group.add_argument('--target_pipeline_parallel_size', type=int,
                       help='Target tensor model parallel size, default to the pipeline parall size '
                       'in the input checkpoint if provided by the loader, otherwise to 1')


def save_checkpoint(queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        import megatron.arguments
        from megatron.checkpointing import save_checkpoint
        from megatron.global_vars import set_global_variables, get_args
        from megatron.model.enums import PositionEmbeddingType
        from megatron.model import ModelType
        from megatron.tokenizer.tokenizer import _vocab_size_with_padding
        from megatron import fused_kernels
        from megatron.core import mpu
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron_path. Exiting.")
        exit(1)

    def queue_get(name=None):
        val = queue.get()
        if val == "exit":
            print("Loader exited, exiting saver")
            exit(1)
        if name is not None and args.checking and val["name"] != name:
            val_name = val["name"]
            print(f'Unexpected message. Expecting "{name}" but got "{val_name}". Exiting saver.')
            exit(1)
        if name is not None:
            print(f"received {name}")
        return val

    def check_message(msg):
        if not args.checking:
            return
        msg_name = msg.pop("name")
        if len(msg.keys()) > 0:
            print(f"Unexpected values in {msg_name}:")
            for key in msg.keys():
                print(f"   {key}")
            print(f"Exiting. If you want to ignore this, use the argument --no_checking.")
            exit(1)

    md = queue_get()

    if args.target_tensor_parallel_size is None:
        if hasattr(md, 'previous_tensor_parallel_size'):
            args.target_tensor_parallel_size = md.previous_tensor_parallel_size
        else:
            print("loader did not provide a tensor parallel size and --target_tensor_parallel_size not provided on command line. "
                  "Default to 1.")
            args.target_tensor_parallel_size = 1

    if args.target_pipeline_parallel_size is None:
        if hasattr(md, 'previous_pipeline_parallel_size'):
            args.target_pipeline_parallel_size = md.previous_pipeline_parallel_size
        else:
            print("loader did not provide a pipeline parallel size and --target_pipeline_parallel_size not provided on command line. "
                  "Default to 1.")
            args.target_pipeline_parallel_size = 1

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    if args.target_tensor_parallel_size is not None and args.target_pipeline_parallel_size is not None:
        os.environ["WORLD_SIZE"] = f'{args.target_tensor_parallel_size * args.target_pipeline_parallel_size}'

    # We want all arguments to come from us
    sys.argv = ['script.py',
                '--num_layers', str(md.num_layers),
                '--hidden_size', str(md.hidden_size),
                '--seq_length', str(md.seq_length),
                '--num_attention_heads', str(md.num_attention_heads),
                '--max_position_embeddings', str(md.max_position_embeddings),
                '--tokenizer_type', str(md.tokenizer_type),
                '--tensor_model_parallel_size', str(args.target_tensor_parallel_size),
                '--pipeline_model_parallel_size', str(args.target_pipeline_parallel_size),
                '--no_masked_softmax_fusion',
                '--no_bias_gelu_fusion',
                '--no_bias_dropout_fusion',
                '--use_cpu_initialization',
                '--micro_batch_size', '1',
                '--no_load_optim',
                '--no_load_rng',
                '--no_save_optim',
                '--no_save_rng',
                '--no_initialization',
                '--save_interval', '1',
                '--hidden_dropout', str(md.hidden_dropout),
                '--position_embedding_type', str(md.position_embedding_type),
                '--save', args.save_dir,
                '--ffn_hidden_size', str(md.ffn_hidden_size)
                ]
    if md.num_attention_heads_kv is not None:
        sys.argv += ["--num_attention_heads_kv", str(md.num_attention_heads_kv)]
    if md.parallel_attn:
        sys.argv += ["--parallel_attn"]
    if md.parallel_layernorm:
        sys.argv += ["--parallel_layernorm"]
    if md.use_flash_attn:
        sys.argv += ["--use_flash_attn"]
    if md.glu_activation is not None:
        sys.argv += ["--glu_activation", str(md.glu_activation)]
    if md.use_rms_norm:
        sys.argv += ["--use_rms_norm"]
    if not md.tie_embed_logits:
        sys.argv += ["--no_tie_embed_logits"]
    if md.lima_dropout:
        sys.argv += ["--lima_dropout"]

    if md.make_vocab_size_divisible_by is not None:
        sys.argv.extend(['--make_vocab_size_divisible_by', str(md.make_vocab_size_divisible_by)])
    if md.params_dtype == torch.float16:
        sys.argv.append('--fp16')
    elif md.params_dtype == torch.bfloat16:
        sys.argv.append('--bf16')

    margs = megatron.arguments.parse_args()
    megatron.arguments.validate_args(margs)
    set_global_variables(margs)
    margs = get_args()

    if hasattr(md, 'consumed_train_samples'):
        margs.consumed_train_samples = md.consumed_train_samples
        margs.consumed_valid_samples = md.consumed_valid_samples
        print(f"Setting consumed_train_samples to {margs.consumed_train_samples}"
              f" and consumed_valid_samples to {margs.consumed_valid_samples}")
    else:
        print("consumed_train_samples not provided.")

    # Determine how to make our models
    if md.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif md.model_type in {'falcon', 'llama', 'llama2', 'codellama', 'mistral'}:
        from finetune import model_provider
        margs.model_name = args.model_type
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    def _get_models(count, dtype, pre_process, post_process):
        models = [model_provider(pre_process, post_process).to(dtype) for _ in range(count)]
        return models

    # fake initializing distributed
    mpu._DATA_PARALLEL_GROUP = 0
    mpu.set_tensor_model_parallel_world_size(args.target_tensor_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(args.target_pipeline_parallel_size)
    mpu.set_tensor_model_parallel_rank(0)
    mpu.set_pipeline_model_parallel_rank(0)
    fused_kernels.load(margs)

    # Embeddings
    embeddings_msg = queue_get("embeddings")

    if md.position_embedding_type == PositionEmbeddingType.absolute:
        pos_embed = embeddings_msg.pop("position embeddings")
    else:
        pos_embed = None
    orig_word_embed = embeddings_msg.pop("word embeddings")
    check_message(embeddings_msg)

    # Get lm_head, if available
    if not md.tie_embed_logits:
        lm_head = queue_get("lm_head").pop("lm_head")

    # Deal with padding
    if md.true_vocab_size is not None:
        # figure out what our padded vocab size is
        orig_vocab_size = orig_word_embed.shape[0]
        margs.padded_vocab_size = _vocab_size_with_padding(md.true_vocab_size, margs)

        # Cut out extra padding we don't need
        if orig_vocab_size > margs.padded_vocab_size:
            full_word_embed = orig_word_embed[0:margs.padded_vocab_size,:]
            if not md.tie_embed_logits:
                full_lm_head = lm_head[:margs.padded_vocab_size, :]

        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < margs.padded_vocab_size:
            padding_size = margs.padded_vocab_size - orig_vocab_size

            full_word_embed = torch.cat((
                orig_word_embed,
                orig_word_embed[-1].unsqueeze(0).expand(padding_size, -1)))

            if not md.tie_embed_logits:
                full_lm_head = torch.cat([
                    lm_head, lm_head[-1].unsqueeze(0).expand(padding_size, -1)
                ])

        # Same size!
        else:
            full_word_embed = orig_word_embed
            if not md.tie_embed_logits:
                full_lm_head = lm_head
    else:
        print("Original vocab size not specified, leaving embedding table as-is. "
              "If you've changed the tensor parallel size this could cause problems.")
        margs.padded_vocab_size = orig_word_embed.shape[0]
        full_word_embed = orig_word_embed
        if not md.tie_embed_logits:
            full_lm_head = lm_head

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_parallel_size, dim=0)
    if not md.tie_embed_logits:
        out_lm_head = torch.chunk(full_lm_head, args.target_tensor_parallel_size, dim=0)

    # Make models for first pipeline stage and fill in embeddings
    mpu.set_pipeline_model_parallel_rank(0)
    post_process = args.target_pipeline_parallel_size == 1
    models = _get_models(args.target_tensor_parallel_size, md.params_dtype, True, post_process)
    models_init = models
    for tp_rank, model in enumerate(models):
        print(f"word embeddings shape {model.language_model.embedding.word_embeddings.weight.shape}")
        model.language_model.embedding.word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
        if pos_embed is not None:
            model.language_model.embedding.position_embeddings.weight.data.copy_(pos_embed)

    # Make models for last pipeline stage and fill in lm_head, if necessary
    if not md.tie_embed_logits:
        mpu.set_pipeline_model_parallel_rank(args.target_pipeline_parallel_size - 1)
        pre_process = args.target_pipeline_parallel_size == 1
        if pre_process:
            models = models_init
        else:
            models = _get_models(args.target_tensor_parallel_size, md.params_dtype, pre_process, True)
        models_final = models
        for tp_rank, model in enumerate(models):
            print(f"lm_head shape {model.language_model.lm_head.shape}")
            model.language_model.lm_head.data.copy_(out_lm_head[tp_rank])

    # Transformer layers
    total_layer_num = 0
    for pp_rank in range(args.target_pipeline_parallel_size):
        # For later pipeline parallel ranks, make the new models
        mpu.set_pipeline_model_parallel_rank(pp_rank)
        post_process = pp_rank == args.target_pipeline_parallel_size - 1
        if pp_rank == 0:
            models = models_init
        elif pp_rank == args.target_pipeline_parallel_size - 1 and not md.tie_embed_logits:
            models = models_final
        else:
            models = _get_models(args.target_tensor_parallel_size, md.params_dtype, False, post_process)

        for layer in range(len(models[0].language_model.encoder.layers)):
            msg = queue_get(f"transformer layer {total_layer_num}")

            # duplicated tensors
            input_layernorm_weight = msg.pop("input layernorm weight")
            if md.parallel_layernorm:
                mlp_layernorm_weight = msg.pop("mlp layernorm weight")
            if not md.use_rms_norm:
                input_layernorm_bias = msg.pop("input layernorm bias")
                if md.parallel_layernorm:
                    mlp_layernorm_bias = msg.pop("mlp layernorm bias")
            if not md.parallel_attn:
                post_layernorm_weight = msg.pop("post layernorm weight")
                if not md.use_rms_norm:
                    post_layernorm_bias = msg.pop("post layernorm bias")
            if md.use_bias:
                dense_bias = msg.pop("dense bias")
                mlp_l1_bias = msg.pop("mlp l1 bias")

            # Split up the parallel tensors
            qkv_weight = torch.chunk(msg.pop("qkv weight"), args.target_tensor_parallel_size, dim=0)
            if md.use_bias:
                qkv_bias = torch.chunk(msg.pop("qkv bias"), args.target_tensor_parallel_size, dim=0)
            dense_weight = torch.chunk(msg.pop("dense weight"), args.target_tensor_parallel_size, dim=1)
            if md.glu_activation is None:
                mlp_l0_weight = torch.chunk(msg.pop("mlp l0 weight"), args.target_tensor_parallel_size, dim=0)
            else:
                up_weight, gate_weight = torch.chunk(msg.pop("mlp l0 weight"), 2, dim=0)
                up_weights = torch.chunk(up_weight, args.target_tensor_parallel_size, dim=0)
                gate_weights = torch.chunk(gate_weight, args.target_tensor_parallel_size, dim=0)
                mlp_l0_weight = [torch.cat([up_weight, gate_weight], dim=0)
                                 for up_weight, gate_weight in zip(up_weights, gate_weights)]
            if md.use_bias:
                mlp_l0_bias = torch.chunk(msg.pop("mlp l0 bias"), args.target_tensor_parallel_size, dim=0)
            mlp_l1_weight = torch.chunk(msg.pop("mlp l1 weight"), args.target_tensor_parallel_size, dim=1)

            # Save them to the model
            for tp_rank in range(args.target_tensor_parallel_size):
                l = models[tp_rank].language_model.encoder.layers[layer]
                l.input_layernorm.weight.data.copy_(input_layernorm_weight)
                if md.parallel_layernorm:
                    l.mlp_layernorm.weight.data.copy_(mlp_layernorm_weight)
                if not md.use_rms_norm:
                    l.input_layernorm.bias.data.copy_(input_layernorm_bias)
                    if md.parallel_layernorm:
                        l.mlp_layernorm.bias.data.copy_(mlp_layernorm_bias)
                l.self_attention.query_key_value.weight.data.copy_(qkv_weight[tp_rank])
                l.self_attention.dense.weight.data.copy_(dense_weight[tp_rank])
                if md.use_bias:
                    l.self_attention.query_key_value.bias.data.copy_(qkv_bias[tp_rank])
                    l.self_attention.dense.bias.data.copy_(dense_bias)
                if not md.parallel_attn:
                    l.post_attention_layernorm.weight.data.copy_(post_layernorm_weight)
                    if not md.use_rms_norm:
                        l.post_attention_layernorm.bias.data.copy_(post_layernorm_bias)
                l.mlp.dense_h_to_4h.weight.data.copy_(mlp_l0_weight[tp_rank])
                l.mlp.dense_4h_to_h.weight.data.copy_(mlp_l1_weight[tp_rank])
                if md.use_bias:
                    l.mlp.dense_h_to_4h.bias.data.copy_(mlp_l0_bias[tp_rank])
                    l.mlp.dense_4h_to_h.bias.data.copy_(mlp_l1_bias)
            total_layer_num = total_layer_num + 1
            check_message(msg)

        if post_process:
            msg = queue_get("final layernorm")
            final_layernorm_weight = msg.pop("weight")
            if not md.use_rms_norm:
                final_layernorm_bias = msg.pop("bias")
            for tp_rank in range(args.target_tensor_parallel_size):
                models[tp_rank].language_model.encoder.final_layernorm.weight.data.copy_(final_layernorm_weight)
                if not md.use_rms_norm:
                    models[tp_rank].language_model.encoder.final_layernorm.bias.data.copy_(final_layernorm_bias)
                if pp_rank != 0 and md.tie_embed_logits:
                    # Copy word embeddings to final pipeline rank
                    models[tp_rank].word_embeddings.weight.data.copy_(out_word_embed[tp_rank])
            del final_layernorm_weight
            if not md.use_rms_norm:
                del final_layernorm_bias
            check_message(msg)

            msg = queue_get()
            if msg != "done" and msg["name"] == "pooler":
                if not hasattr(models[0].language_model, 'pooler'):
                    print("ERROR: got a pooler, but model does not have one")
                    exit(1)
                print("received pooler")
                pooler_weight = msg.pop("weight")
                pooler_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].language_model.pooler.dense.weight.data.copy_(pooler_weight)
                    models[tp_rank].language_model.pooler.dense.bias.data.copy_(pooler_bias)
                del pooler_weight
                del pooler_bias
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "lm head":
                if not hasattr(models[0], 'lm_head'):
                    print("ERROR: got an lm head, but model does not have one")
                    exit(1)
                print("received lm head")
                lm_head_dense_weight = msg.pop("dense weight")
                lm_head_dense_bias = msg.pop("dense bias")
                lm_head_layernorm_weight = msg.pop("layernorm weight")
                lm_head_layernorm_bias = msg.pop("layernorm bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].lm_head.dense.weight.data.copy_(lm_head_dense_weight)
                    models[tp_rank].lm_head.dense.bias.data.copy_(lm_head_dense_bias)
                    models[tp_rank].lm_head.layernorm.weight.data.copy_(lm_head_layernorm_weight)
                    models[tp_rank].lm_head.layernorm.bias.data.copy_(lm_head_layernorm_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done" and msg["name"] == "binary head":
                if not hasattr(models[0], 'binary_head'):
                    print("ERROR: got a binary head, but model does not have one")
                    exit(1)
                print("received binary head")
                binary_head_weight = msg.pop("weight")
                binary_head_bias = msg.pop("bias")
                for tp_rank in range(args.target_tensor_parallel_size):
                    models[tp_rank].binary_head.weight.data.copy_(binary_head_weight)
                    models[tp_rank].binary_head.bias.data.copy_(binary_head_bias)
                check_message(msg)
                msg = queue_get()

            if msg != "done":
                print("ERROR: got some more data but was expecting to be done")

        for tp_rank in range(args.target_tensor_parallel_size):
            mpu.set_tensor_model_parallel_rank(tp_rank)
            save_checkpoint(md.iteration, [models[tp_rank]], None, None)
    print("Done!")
