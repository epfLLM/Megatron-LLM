import json
import os
import sys
import types

import torch


def add_arguments(parser):
    group = parser.add_argument_group(title='Megatron loader')

    group.add_argument('--true_vocab_size', type=int, default=None,
                       help='original size of vocab, if specified will trim padding from embedding table.')
    group.add_argument('--vocab_file', type=str, default=None,
                       help='Path to the vocab file. If specified will use this to get vocab size and '
                       'trim padding from the embedding table.')
    group.add_argument('--megatron_path', type=str, default=None,
                       help='Base directory of deepspeed repository')


def _load_checkpoint(queue, args):
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    try:
        import megatron.arguments
        from megatron.global_vars import set_global_variables
        from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.model.enums import PositionEmbeddingType
        from megatron.model import ModelType, module
        from megatron.core import mpu
        from megatron import fused_kernels
    except ModuleNotFoundError:
        print("Unable to import Megatron, please specify the path to Megatron using --megatron_path. Exiting.")
        queue.put("exit")
        exit(1)

    # We want all arguments to come from us
    sys.argv = ['script.py',
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
                '--load', args.load_dir
                ]

    if args.bf16:
        sys.argv += ["--bf16"]

    margs = megatron.arguments.parse_args()

    if args.load_iters is not None:
        margs.load_iters = args.load_iters

    margs = load_args_from_checkpoint(margs)

    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    margs.world_size = margs.tensor_model_parallel_size * margs.pipeline_model_parallel_size

    margs = megatron.arguments.validate_args(margs)

    def check_for_arg(arg_name):
        if getattr(margs, arg_name, None) is None:
            print(f"Checkpoint does not specify the argument {arg_name}. Exiting.")
            print(f"Arguments: {margs}")
            queue.put("exit")
            exit(1)

    check_for_arg('tensor_model_parallel_size')
    check_for_arg('pipeline_model_parallel_size')
    check_for_arg('num_layers')
    check_for_arg('hidden_size')
    check_for_arg('seq_length')
    check_for_arg('num_attention_heads')
    check_for_arg('max_position_embeddings')
    check_for_arg('tokenizer_type')
    check_for_arg('iteration')
    check_for_arg('params_dtype')
    if args.model_type == "BERT":
        check_for_arg('bert_binary_head')

    # Determine how to make our models
    if args.model_type == 'GPT':
        from pretrain_gpt import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type in {"falcon", "llama", "llama2", "codellama"}:
        from finetune import model_provider
        margs.model_name = args.model_type
        margs.model_type = ModelType.encoder_or_decoder
    elif args.model_type == 'BERT':
        from pretrain_bert import model_provider
        margs.model_type = ModelType.encoder_or_decoder
    else:
        raise Exception(f'unrecognized model type: {args.model_type}')

    # supress warning about torch.distributed not being initialized
    module.MegatronModule.embedding_warning_printed = True

    consumed_train_samples = None
    consumed_valid_samples = None

    def _get_models(count, dtype, pre_process, post_process):
        nonlocal consumed_train_samples
        nonlocal consumed_valid_samples
        models = []
        for rank in range(count):
            mpu.set_tensor_model_parallel_rank(rank)
            model_ = [model_provider(pre_process, post_process).to(dtype)]
            margs.consumed_train_samples = 0
            margs.consumed_valid_samples = 0
            load_checkpoint(model_, None, None)
            assert(len(model_) == 1)
            model_ = model_[0]
            if consumed_train_samples is not None:
                assert(margs.consumed_train_samples == consumed_train_samples)
            else:
                consumed_train_samples = margs.consumed_train_samples
            if consumed_valid_samples is not None:
                assert(margs.consumed_valid_samples == consumed_valid_samples)
            else:
                consumed_valid_samples = margs.consumed_valid_samples
            models.append(model_)
        return models

    if margs.num_layers_per_virtual_pipeline_stage is not None:
        print("Model with an interleaved pipeline schedule are not yet supported.")
        queue.put("exit")
        exit(1)

    set_global_variables(margs)
    mpu._DATA_PARALLEL_GROUP = 0
    mpu.set_tensor_model_parallel_world_size(margs.tensor_model_parallel_size)
    mpu.set_pipeline_model_parallel_world_size(margs.pipeline_model_parallel_size)
    fused_kernels.load(margs)

    # Get true (non-padded) vocab size
    if args.true_vocab_size is not None:
        true_vocab_size = args.true_vocab_size
    elif args.vocab_file is not None:
        vocab = json.load(open(args.vocab_file))
        true_vocab_size = len(vocab)
        if args.true_vocab_size is not None and true_vocab_size != args.true_vocab_size:
            print("Both --true_vocab_size and --vocab_file specified and the vocab size does not match, aborting.")
            queue.put("exit")
            exit(1)
    else:
        true_vocab_size = None

    # short aliases
    tp_size = margs.tensor_model_parallel_size
    pp_size = margs.pipeline_model_parallel_size

    # metadata
    md = types.SimpleNamespace()
    md.model_type = args.model_type
    md.num_layers = margs.num_layers
    md.hidden_size = margs.hidden_size
    md.seq_length = margs.seq_length
    md.num_attention_heads = margs.num_attention_heads
    md.max_position_embeddings = margs.max_position_embeddings
    md.tokenizer_type = margs.tokenizer_type
    md.iteration = margs.iteration
    if args.model_type == "BERT":
        md.bert_binary_head = margs.bert_binary_head
    md.previous_tensor_parallel_size = margs.tensor_model_parallel_size
    md.previous_pipeline_parallel_size = margs.pipeline_model_parallel_size
    md.true_vocab_size = true_vocab_size
    md.make_vocab_size_divisible_by = margs.make_vocab_size_divisible_by
    md.num_attention_heads_kv = margs.num_attention_heads_kv
    md.parallel_attn = margs.parallel_attn
    md.parallel_layernorm = margs.parallel_layernorm
    md.use_flash_attn = margs.use_flash_attn
    md.hidden_dropout = margs.hidden_dropout
    md.lima_dropout = margs.lima_dropout
    md.use_bias = margs.use_bias
    md.use_rms_norm = margs.use_rms_norm
    md.ffn_hidden_size = margs.ffn_hidden_size
    md.glu_activation = margs.glu_activation
    md.tie_embed_logits = margs.tie_embed_logits
    md.params_dtype = margs.params_dtype
    if margs.position_embedding_type == PositionEmbeddingType.absolute:
        md.position_embedding_type = "absolute"
    elif margs.position_embedding_type == PositionEmbeddingType.rotary:
        md.position_embedding_type = "rotary"
    else:
        raise KeyError(f"Unknown position embedding {margs.position_embedding_type}")

    # Get first pipe stage
    mpu.set_pipeline_model_parallel_rank(0)
    post_process = pp_size == 1
    models = _get_models(tp_size, md.params_dtype, True, post_process)
    models_init = models

    md.consumed_train_samples = consumed_train_samples
    md.consumed_valid_samples = consumed_valid_samples
    queue.put(md)

    def queue_put(name, msg):
        print(f"sending {name}")
        msg["name"] = name
        queue.put(msg)

    message = {
        "word embeddings": torch.cat(
            [models[tp_rank].language_model.embedding.word_embeddings.weight.data for tp_rank in range(tp_size)],
            dim = 0)
    }
    if margs.position_embedding_type == PositionEmbeddingType.absolute:
        message["position embeddings"] = models[0].language_model.embedding.position_embeddings.weight.data

    queue_put("embeddings", message)

    # Get last pipe stage if lm_head needs to be sent
    if not margs.tie_embed_logits:
        mpu.set_pipeline_model_parallel_rank(pp_size - 1)
        pre_process = pp_size == 1
        if pre_process:
            models = models_init
        else:
            models = _get_models(tp_size, md.params_dtype, pre_process, True)
        models_final = models

        queue_put("lm_head", {"lm_head": torch.cat([models[tp_rank].language_model.lm_head.data
                                                    for tp_rank in range(tp_size)])})

    total_layer_num = 0
    for pp_rank in range(pp_size):
        # For later pipeline parallel ranks, make the new models
        mpu.set_pipeline_model_parallel_rank(pp_rank)
        post_process = pp_rank == pp_size - 1
        if pp_rank == 0:
            models = models_init
        elif pp_rank == pp_size - 1 and not md.tie_embed_logits:
            models = models_final
        else:
            models = _get_models(tp_size, md.params_dtype, False, post_process)

        for layer_num in range(len(models[0].language_model.encoder.layers)):
            message = {}

            # Get non-parallel tensors from tp_rank 0
            layer = models[0].language_model.encoder.layers[layer_num]
            message["input layernorm weight"] = layer.input_layernorm.weight.data
            if margs.parallel_layernorm:
                message["mlp layernorm weight"] = layer.mlp_layernorm.weight.data
            if not margs.use_rms_norm:
                message["input layernorm bias"] = layer.input_layernorm.bias.data
                if margs.parallel_layernorm:
                    message["mlp layernorm bias"] = layer.mlp_layernorm.bias.data
            if not margs.parallel_attn:
                message["post layernorm weight"] = layer.post_attention_layernorm.weight.data
                if not margs.use_rms_norm:
                    message["post layernorm bias"] = layer.post_attention_layernorm.bias.data
            if margs.use_bias:
                message["dense bias"] = layer.self_attention.dense.bias.data
                message["mlp l1 bias"] = layer.mlp.dense_4h_to_h.bias.data

            # Grab all parallel tensors for this layer
            qkv_weight = []
            qkv_bias = []
            dense_weight = []
            mlp_l0_weight = []
            mlp_l0_bias = []
            mlp_l1_weight = []
            for tp_rank, model in enumerate(models):
                layer = model.language_model.encoder.layers[layer_num]
                qkv_weight.append(layer.self_attention.query_key_value.weight.data)
                if margs.use_bias:
                    qkv_bias.append(layer.self_attention.query_key_value.bias.data)
                dense_weight.append(layer.self_attention.dense.weight.data)
                mlp_l0_weight.append(layer.mlp.dense_h_to_4h.weight.data)
                if margs.use_bias:
                    mlp_l0_bias.append(layer.mlp.dense_h_to_4h.bias.data)
                mlp_l1_weight.append(layer.mlp.dense_4h_to_h.weight.data)

            # concat them
            message["qkv weight"] = torch.cat(qkv_weight, dim=0)
            if margs.use_bias:
                message["qkv bias"] = torch.cat(qkv_bias, dim=0)
            message["dense weight"] = torch.cat(dense_weight, dim=1)
            if margs.glu_activation is None:
                message["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)
            else:
                up_weights = []
                gate_weights = []
                for weight in mlp_l0_weight:
                    up, gate = torch.chunk(weight, 2, dim=0)
                    up_weights.append(up)
                    gate_weights.append(gate)
                message["mlp l0 weight"] = torch.cat(up_weights + gate_weights, dim=0)
            if margs.use_bias:
                message["mlp l0 bias"] = torch.cat(mlp_l0_bias, dim=0)
            message["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)

            queue_put(f"transformer layer {total_layer_num}", message)

            total_layer_num = total_layer_num + 1

    # Send final layernorm from tp_rank 0
    message = {
        "weight": models[0].language_model.encoder.final_layernorm.weight.data
    }
    if not margs.use_rms_norm:
        message["bias"] = models[0].language_model.encoder.final_layernorm.bias.data
    queue_put("final layernorm", message)

    # Send BERT lm head and binary head if it exists
    if md.model_type == 'BERT':
        message = {
            "weight": models[0].language_model.pooler.dense.weight.data,
            "bias": models[0].language_model.pooler.dense.bias.data
        }
        queue_put("pooler", message)

        message = {
            "dense weight": models[0].lm_head.dense.weight.data,
            "dense bias": models[0].lm_head.dense.bias.data,
            "layernorm weight": models[0].lm_head.layernorm.weight.data,
            "layernorm bias": models[0].lm_head.layernorm.bias.data
        }
        queue_put("lm head", message)

        if args.model_type == "BERT" and md.bert_binary_head:
            print("Sending BERT Binary head")
            queue.put("binary head")
            message = {
                "weight": models[0].binary_head.weight.data,
                "bias": models[0].binary_head.bias.data
            }
            queue_put("binary head", message)
    queue.put("done")


def load_checkpoint(queue, args):
    try:
        _load_checkpoint(queue, args)
    except:
        queue.put("exit")
        raise
