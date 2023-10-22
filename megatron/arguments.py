# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron arguments."""

import argparse
import os

import torch

import megatron
from megatron.metrics import METRICS
from megatron.model.enums import PositionEmbeddingType


def build_base_parser():
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)
    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_biencoder_args(parser)
    parser = _add_vision_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_inference_args(parser)
    parser = _add_transformer_engine_args(parser)
    return parser


def parse_args(extra_args_provider=None):
    """Parse all arguments."""
    parser = build_base_parser()
    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    args = parser.parse_args()

    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    return args


def validate_args(args, defaults={}):
    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size)
    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))
    args.transformer_pipeline_model_parallel_size = (
        args.pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.pipeline_model_parallel_size
    )
    # Checks.
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    assert args.world_size % model_parallel_size == 0, 'world size is not'\
        ' divisible by tensor parallel size ({}) times pipeline parallel ' \
        'size ({})'.format(args.world_size, args.tensor_model_parallel_size,
                           args.pipeline_model_parallel_size)
    args.data_parallel_size = args.world_size // model_parallel_size
    if args.rank == 0:
        print('using world size: {}, data-parallel-size: {}, '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)
    if args.pipeline_model_parallel_size > 1:
        if args.pipeline_model_parallel_split_rank is not None:
            assert args.pipeline_model_parallel_split_rank < \
                    args.pipeline_model_parallel_size, 'split rank needs'\
                    ' to be less than pipeline model parallel size ({})'.format(
                            args.pipeline_model_parallel_size)

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations
    if args.metrics == ["all"]:
        args.metrics = list(METRICS)


    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    if args.num_layers_per_virtual_pipeline_stage is not None:
        assert args.pipeline_model_parallel_size > 2, \
            'pipeline-model-parallel size should be greater than 2 with ' \
            'interleaved schedule'
        assert args.num_layers % args.num_layers_per_virtual_pipeline_stage == 0, \
            'number of layers is not divisible by number of layers per virtual ' \
            'pipeline stage'
        args.virtual_pipeline_model_parallel_size = \
            (args.num_layers // args.transformer_pipeline_model_parallel_size) // \
            args.num_layers_per_virtual_pipeline_stage
    else:
        args.virtual_pipeline_model_parallel_size = None

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if not args.accumulate_allreduce_grads_in_fp32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    # If we do accumulation and all-reduces in fp32, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is not off.
    if args.accumulate_allreduce_grads_in_fp32:
        assert args.DDP_impl == 'local'
        assert args.use_contiguous_buffers_in_local_ddp

    # If we use the distributed optimizer, we need to have local DDP
    # and we should make sure use-contiguous-buffers-in-local-ddp is on.
    if args.use_distributed_optimizer:
        assert args.DDP_impl == 'local'
        assert args.use_contiguous_buffers_in_local_ddp

    # For torch DDP, we do not use contiguous buffer
    if args.DDP_impl == 'torch':
        args.use_contiguous_buffers_in_local_ddp = False

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    if args.variable_seq_lengths is None:
        args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr_warmup_fraction and lr_warmup_iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learning rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr_warmup_fraction ' \
                'and lr_warmup_samples'

    if args.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num_layers and encoder_num_layers specified'
        args.encoder_num_layers = args.num_layers
    else:
        assert args.encoder_num_layers is not None, \
            'either num_layers or encoder_num_layers should be specified'
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    # required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
    #                  'max_position_embeddings']
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None:
        assert args.hidden_size % args.num_attention_heads == 0
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.num_attention_heads_kv is None:
        args.num_attention_heads_kv = args.num_attention_heads

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if not isinstance(args.position_embedding_type, PositionEmbeddingType):
        args.position_embedding_type = PositionEmbeddingType[args.position_embedding_type]
    if args.position_embedding_type in [PositionEmbeddingType.absolute, PositionEmbeddingType.rotary]:
        assert args.max_position_embeddings is not None
        if args.seq_length is not None:
            assert args.max_position_embeddings >= args.seq_length
        if args.decoder_seq_length is not None:
            assert args.max_position_embeddings >= args.decoder_seq_length
        assert args.rope_scaling_factor >= 1, 'rope_scaling_factor must be >= 1'
    else:
        assert args.max_position_embeddings is None

    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.weight_decay_incr_style == 'constant':
        assert args.start_weight_decay is None
        assert args.end_weight_decay is None
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        assert args.start_weight_decay is not None
        assert args.end_weight_decay is not None

    TORCH_MAJOR = int(torch.__version__.split('.')[0])
    TORCH_MINOR = int(torch.__version__.split('.')[1])
    # Persistent fused layer norm.
    if TORCH_MAJOR < 1 or (TORCH_MAJOR == 1 and TORCH_MINOR < 11):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation recomputing.
    if args.distribute_saved_activations:
        assert args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert TORCH_MAJOR >= 1 and TORCH_MINOR >= 10, \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            'pytorch version is v%s.%s.' % (TORCH_MAJOR, TORCH_MINOR)

    # Tranformer-Engine/FP8 related checking
    if args.fp8_e4m3 or args.fp8_hybrid:
        assert args.transformer_impl == 'transformer_engine', \
            'transformer-engine required for fp8 training and inference'

    assert not (args.fp8_e4m3 and args.fp8_hybrid), \
        'cannot train with both fp8 e4m3 and hybrid formatting'

    if args.fp16:
        assert args.transformer_impl == 'local', \
            'transformer-engine not yet approved for fp16 training and inference'

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    # Parallel attention.
    if not args.parallel_attn:
        assert not args.parallel_layernorm, "parallel_layernorm only implemented with parallel_attention"

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        args.sequence_parallel = False

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False

    if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') and os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
        if args.sequence_parallel:
            raise RuntimeError(
                "Using sequence parallelism requires setting the environment variable "
                "CUDA_DEVICE_MAX_CONNECTIONS to 1")
        if args.async_tensor_model_parallel_allreduce:
            raise RuntimeError(
                "Using async gradient all reduce requires setting the environment "
                "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")
    _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('------------------------ arguments ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('-------------------- end of arguments ---------------------',
              flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')
    group.add_argument('--fp8_e4m3', action='store_true',
                        help='E4M3 TransformerLayer', dest='fp8_e4m3')
    group.add_argument('--fp8_hybrid', action='store_true',
                        help='Hybrid FP8 TransformerLayer')
    group.add_argument('--no_fp8_wgrad', action='store_false',
                        help='Execute wgrad in higher precision even for FP8 runs', dest='fp8_wgrad')
    group.add_argument('--fp8_margin', type=int, default=0,
                        help='Scaling margin for fp8', dest='fp8_margin')
    group.add_argument('--fp8_interval', type=int, default=1,
                        help='Scaling update interval for fp8', dest='fp8_interval')
    group.add_argument('--transformer_impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    group.add_argument('--fp8_amax_history_len', type=int, default=1,
                        help='Number of steps for which amax history is recorded per tensor')
    group.add_argument('--fp8_amax_compute_algo', default='most_recent',
                       choices=['most_recent', 'max'],
                       help='Algorithm for computing amax from history')
    return parser


def _add_inference_args(parser):
    group = parser.add_argument_group(title='inference')
    group.add_argument('--inference_batch_times_seqlen_threshold',
                       type=int, default=512,
                       help='During inference, if batch-size times '
                       'sequence-length is smaller than this threshold '
                       'then we will not use pipelining, otherwise we will.')
    group.add_argument('--max_tokens_to_oom',
                       type=int, default=12000,
                       help='Maximum number of tokens during inference'
                       'tokens here is # in prompt + # to generate'
                       'Allows us to throw an error before OOM crashes server')
    return parser

    
def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')
    group.add_argument('--num_layers', type=int, default=None,
                       help='Number of transformer layers.')
    group.add_argument('--encoder_num_layers', type=int, default=None,
                       help='Number of encoder transformer layers.')
    group.add_argument('--decoder_num_layers', type=int, default=None,
                       help='Number of decoder transformer layers.')
    group.add_argument('--hidden_size', type=int, default=None,
                       help='Tansformer hidden size.')
    group.add_argument('--ffn_hidden_size', type=int, default=None,
                       help='Transformer Feed-Forward Network hidden size. '
                       'This is set to 4*hidden_size if not provided')
    group.add_argument('--num_attention_heads', type=int, default=None,
                       help='Number of transformer attention heads.')
    group.add_argument('--num_attention_heads_kv', type=int, default=None,
                       help='Number of transformer attention heads for the keys and values.')
    group.add_argument('--kv_channels', type=int, default=None,
                       help='Projection weights dimension in multi-head '
                       'attention. This is set to '
                       '   args.hidden_size // args.num_attention_heads '
                       'if not provided.')
    group.add_argument('--max_position_embeddings', type=int, default=None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--make_vocab_size_divisible_by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--layernorm_epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    group.add_argument('--apply_residual_connection_post_layernorm',
                       action='store_true',
                       help='If set, use original BERT residual connection '
                       'ordering.')
    group.add_argument('--use_bias', action='store_true',
                       help='If set then use bias.')  # Added during hackathon
    # Extracted from: https://github.com/facebookresearch/llama/blob/main/llama/model.py
    group.add_argument('--use_rms_norm',
                       action='store_true',
                       help='If set, use RMSNorm instead of LayerNorm.')
    group.add_argument('--use_post_ln',
                       action='store_true',
                       help='If set, use Post-LN transformer (in the notation of https://sh-tsang.medium.com/review-pre-ln-transformer-on-layer-normalization-in-the-transformer-architecture-b6c91a89e9ab).')
    group.add_argument('--onnx_safe', type=bool, required=False,
                       help='Use workarounds for known problems with '
                       'Torch ONNX exporter')
    # Extracted from: https://github.com/bigscience-workshop/Megatron-DeepSpeed
    group.add_argument('--glu_activation', type=str,
                       choices=megatron.model.glu_activations.GLU_ACTIVATIONS.keys(),
                       help='GLU activations to use.'
                       )
    group.add_argument('--position_embedding_type', type=lambda x: PositionEmbeddingType[x],
                       choices=list(PositionEmbeddingType),
                       default=PositionEmbeddingType.absolute,
                       help='Define position embedding type ("absolute" | "rotary"). "absolute" by default.')
    group.add_argument('--rope_scaling_factor', type=float, default=1.0,
                       help='Set the linear RoPE scaling factor for sequence interpolation.')
    group.add_argument('--rope_theta', type=float, default=10000.0,
                       help='Set RoPE theta base (llama/llama2: 1e4, codellama: 1e6).')
    # Added mainly for Falcon
    group.add_argument("--parallel_attn", action="store_true",
                       help="Whether to use parallel mlp and attn computation with a single layernorm")
    group.add_argument("--parallel_layernorm", action="store_true",
                       help="Whether to use a dedicated layernorm for the mlp in the attention")
    # Added mainly for Llama
    group.add_argument("--no_tie_embed_logits", action="store_false", dest="tie_embed_logits",
                       help=("If set, the weights of the word embedding and lm_head "
                             "are not tied"))
    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')
    group.add_argument('--log_params_norm', action='store_true',
                       help='If set, calculate and log parameters norm.')
    group.add_argument('--log_num_zeros_in_grad', action='store_true',
                       help='If set, calculate and log the number of zeros in gradient.')
    group.add_argument('--timing_log_level', type=int,
                       default=0, choices=range(0, 3),
                       help='Granularity level to measure and report timing. '
                       '   0: report only iteration time and make sure timing '
                       '      does not introduce extra overhead.'
                       '   1: report timing for operations that are executed '
                       '      very limited times (basically once) during '
                       '      each iteration (such as gradient all-reduce) '
                       '   2: report timing for operations that migh be '
                       '      executed numerous times during each iteration. '
                       'Note that setting the level to 1 or 2 might '
                       'cause increase in iteration time.')
    group.add_argument('--barrier_with_L1_time', action='store_false',
                       help='If not set, use barrier with level 1 time '
                       'measurements. Note that this is up to the user '
                       'to make sure calling barrier with their timers '
                       'will not result in hangs. This can happen if for '
                       'example the user adds a level 1 timer that is not '
                       'called by all ranks.')
    group.add_argument('--timing_log_option', type=str, default='minmax',
                       choices=['max', 'minmax', 'all'],
                       help='Options for logging timing:'
                       '  max: report the max timing across all ranks'
                       '  minmax: report min and max timings across all ranks'
                       '  all: report timings of all ranks.')
    group.add_argument('--tensorboard_log_interval', type=int, default=1,
                       help='Report to tensorboard interval.')
    group.add_argument('--tensorboard_queue_size', type=int, default=1000,
                       help='Size of the tensorboard queue for pending events '
                       'and summaries before one of the ‘add’ calls forces a '
                       'flush to disk.')
    group.add_argument('--log_timers_to_tensorboard', action='store_true',
                       help='If set, write timers to tensorboard.')
    group.add_argument('--log_batch_size_to_tensorboard', action='store_true',
                       help='If set, write batch-size to tensorboard.')
    group.add_argument('--log_validation_ppl_to_tensorboard',
                       action='store_true',
                       help='If set, write validation perplexity to '
                       'tensorboard.')
    group.add_argument('--log_memory_to_tensorboard',
                       action='store_true',
                       help='Enable memory logging to tensorboard.')
    group.add_argument('--log_world_size_to_tensorboard',
                       action='store_true',
                       help='Enable world size logging to tensorboard.')
    group.add_argument('--wandb_logger',
                       action='store_true',
                       help='Enable logging to Weights & Biases instead of tensorboard.')
    group.add_argument('--wandb_project', type=str, default=None,
                       help='Project name for Weights & Biases.')
    group.add_argument('--wandb_entity', type=str, default="meditron",
                       help='Entity/team name for Weights & Biases.')
    group.add_argument('--wandb_id',type=str,default=None,
                       help="Unique ID to identify this run, alternatively can set `WANDB_RUN_ID`.")
    group.add_argument('--wandb_resume',action="store_true",
                       help="If set, we resume logging for the id given instead of launching a new run (errors if id given and resume=False).")
    group.add_argument("--wandb_api_key",type=str,default=None,
                       help="API key for Weights & Biases, needs to be set if not set in environment variable `WANDB_API_KEY`.")
    group.add_argument("--metrics", default=[], nargs="+", choices=list(METRICS) + ["all"],
                       help="Metrics to report when logging")
    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')
    group.add_argument('--attention_dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden_dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    # see "LIMA: Less Is More for Alignment", Zhou et al 2023, https://arxiv.org/abs/2305.11206
    group.add_argument('--lima_dropout', action='store_true',
                       help='Linearly raise the hidden_dropout probability from 0.0 at the first layer to the full hidden_dropout value at the last layer.')
    group.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--start_weight_decay', type=float,
                       help='Initial weight decay coefficient for L2 regularization.')
    group.add_argument('--end_weight_decay', type=float,
                       help='End of run weight decay coefficient for L2 regularization.')
    group.add_argument('--weight_decay_incr_style', type=str, default='constant',
                       choices=['constant', 'linear', 'cosine'],
                       help='Weight decay increment function.')
    group.add_argument('--clip_grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam_beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam_beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages '
                       'of gradient and its square')
    group.add_argument('--adam_eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve'
                       'numerical stability')
    group.add_argument('--sgd_momentum', type=float, default=0.9,
                       help='Momentum factor for sgd')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')
    group.add_argument('--micro_batch_size', type=int, default=None,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    group.add_argument('--global_batch_size', type=int, default=None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro_batch_size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro_batch_size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup_batch_size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup_batch_size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup_batch_size 16 8 300000 \ '
                       '   --global_batch_size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--recompute_activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--recompute_granularity', type=str, default=None,
                       choices=['full', 'selective'],
                       help='Checkpoint activations to allow for training '
                       'with larger models, sequences, and batch sizes. '
                       'It is supported at two granularities 1) full: '
                       'whole transformer layer is recomputed, '
                       '2) selective: core attention part of the transformer '
                       'layer is recomputed.')
    group.add_argument('--distribute_saved_activations',
                       action='store_true',
                       help='If set, distribute recomputed activations '
                       'across model parallel group.')
    group.add_argument('--recompute_method', type=str, default=None,
                       choices=['uniform', 'block'],
                       help='1) uniform: uniformly divide the total number of '
                       'Transformer layers and recompute the input activation of '
                       'each divided chunk at specified granularity, '
                       '2) recompute the input activations of only a set number of '
                       'individual Transformer layers per pipeline stage and do the '
                       'rest without any recomputing at specified granularity'
                       'default) do not apply activations recompute to any layers')
    group.add_argument('--recompute_num_layers', type=int, default=1,
                       help='1) uniform: the number of Transformer layers in each '
                       'uniformly divided recompute unit, '
                       '2) block: the number of individual Transformer layers '
                       'to recompute within each pipeline stage.')
    group.add_argument('--train_iters', type=int, default=None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train_iters or '
                       'train_samples should be provided.')
    group.add_argument('--skip_iters', type=int, nargs='*', default=[],
                        help=('One or more iterations to ignore. Neither the forward '
                              'nor backward pass will be computed for this iterations'))
    group.add_argument('--train_samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train_iters or '
                       'train_samples should be provided.')
    group.add_argument('--log_interval', type=int, default=100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit_interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit_duration_in_mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--exit_signal_handler', action='store_true',
                       help='Dynamically save the checkpoint and shutdown the '
                       'training if SIGTERM is received')
    group.add_argument('--tensorboard_dir', type=str, default=None,
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--no_masked_softmax_fusion',
                       action='store_false',
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='masked_softmax_fusion')
    group.add_argument('--no_bias_gelu_fusion', action='store_false',
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion')
    group.add_argument('--no_bias_dropout_fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')
    group.add_argument('--use_flash_attn', action='store_true',
                       help='use FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'],
                       help='Optimizer function')
    group.add_argument('--dataloader_type', type=str, default=None,
                       choices=['single', 'cyclic'],
                       help='Single pass vs multiple pass data loader')
    group.add_argument('--no_async_tensor_model_parallel_allreduce',
                       action='store_false',
                       help='Disable asynchronous execution of '
                       'tensor-model-parallel all-reduce with weight '
                       'gradient compuation of a column-linear layer.',
                       dest='async_tensor_model_parallel_allreduce')
    group.add_argument('--no_persist_layer_norm', action='store_true',
                       help='Disable using persistent fused layer norm kernel. '
                       'This kernel supports only a set of hidden sizes. Please '
                       'check persist_ln_hidden_sizes if your hidden '
                       'size is supported.')
    group.add_argument('--sequence_parallel', action='store_true',
                       help='Enable sequence parallel optimization.')
    group.add_argument('--no_gradient_accumulation_fusion',
                       action='store_false',
                       help='Disable fusing gradient accumulation to weight '
                       'gradient computation of linear layers',
                       dest='gradient_accumulation_fusion')
    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')
    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--data_parallel_random_init', action='store_true',
                       help='Enable random initialization of params '
                       'across data parallel ranks')
    group.add_argument('--init_method_std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')
    group.add_argument('--init_method_xavier_uniform', action='store_true',
                       help='Enable Xavier uniform parameter initialization')
    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')
    group.add_argument('--lr', type=float, default=None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr_decay_style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'inverse-square-root'],
                       help='Learning rate decay function.')
    group.add_argument('--lr_decay_iters', type=int, default=None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train_iters`')
    group.add_argument('--lr_decay_samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train_samples`')
    group.add_argument('--lr_warmup_fraction', type=float, default=None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr_warmup_iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr_warmup_samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--min_lr', type=float, default=0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override_opt_param_scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use_checkpoint_opt_param_scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')
    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--save', type=str, default=None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save_interval', type=int, default=None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no_save_optim', action='store_true', default=None,
                       help='Do not save current optimizer.')
    group.add_argument('--no_save_rng', action='store_true', default=None,
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--load_iters', type=int, default=None,
                       help='Specify which checkpoint to load. If not '
                          'specified, the latest checkpoint (highest iteration '
                            'number) located in the load directory will be used.')
    group.add_argument('--no_load_optim', action='store_true', default=None,
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no_load_rng', action='store_true', default=None,
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    group.add_argument('--no_initialization', action='store_false',
                       help='Do not perform initialization when building model, '
                       'can reduce startup time when definitely loading from a '
                       'checkpoint',
                       dest='perform_initialization')
    group.add_argument('--use_checkpoint_args', action='store_true',
                       help='Override any command line arguments with arguments '
                       'from the checkpoint')
    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')
    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--bf16', action='store_true',
                       help='Run model in bfloat16 mode.')
    group.add_argument('--loss_scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial_loss_scale', type=float, default=2**32,
                       help='Initial loss scale for dynamic loss scaling.')
    group.add_argument('--min_loss_scale', type=float, default=1.0,
                       help='Minimum loss scale for dynamic loss scale.')
    group.add_argument('--loss_scale_window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--hysteresis', type=int, default=2,
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--fp32_residual_connection', action='store_true',
                       help='Move residual connections to fp32.')
    group.add_argument('--no_query_key_layer_scaling', action='store_false',
                       help='Do not scale Q * K^T by 1 / layer-number.',
                       dest='apply_query_key_layer_scaling')
    group.add_argument('--attention_softmax_in_fp32', action='store_true',
                       help='Run attention masking and softmax in fp32. '
                       'This flag is ignored unless '
                       '--no_query_key_layer_scaling is specified.')
    group.add_argument('--accumulate_allreduce_grads_in_fp32',
                       action='store_true',
                       help='Gradient accumulation and all-reduce in fp32.')
    group.add_argument('--fp16_lm_cross_entropy',
                       action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')
    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')
    group.add_argument('--tensor_model_parallel_size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--pipeline_model_parallel_size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--pipeline_model_parallel_split_rank',
                       type=int, default=None,
                       help='Rank where encoder and decoder should be split.')
    group.add_argument('--num_layers_per_virtual_pipeline_stage', type=int, default=None,
                       help='Number of layers per virtual pipeline stage')
    group.add_argument('--distributed_backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--DDP_impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--no_contiguous_buffers_in_local_ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
    group.add_argument('--no_scatter_gather_tensors_in_pipeline',
                       action='store_false',
                       help='Use scatter/gather to optimize communication of tensors in pipeline',
                       dest='scatter_gather_tensors_in_pipeline')
    group.add_argument('--use_ring_exchange_p2p', action='store_true',
                       default=False, help='If set, use custom-built ring exchange '
                       'for p2p communications. Note that this option will require '
                       'a custom built image that support ring-exchange p2p.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--use_cpu_initialization', action='store_true',
                       default=None, help='If set, affine parallel weights '
                       'initialization uses CPU')
    group.add_argument('--empty_unused_memory_level', default=0, type=int,
                       choices=[0, 1, 2],
                       help='Call torch.cuda.empty_cache() each iteration '
                       '(training and eval), to reduce fragmentation.'
                       '0=off, 1=moderate, 2=aggressive.')
    group.add_argument('--standalone_embedding_stage', action='store_true',
                       default=False, help='If set, *input* embedding layer '
                       'is placed on its own pipeline stage, without any '
                       'transformer layers. (For T5, this flag currently only '
                       'affects the encoder embedding.)')
    group.add_argument('--use_distributed_optimizer', action='store_true',
                       help='Use distributed optimizer.')
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')
    group.add_argument('--eval_iters', type=int, default=100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval_interval', type=int, default=1000,
                       help='Interval between running evaluation on '
                       'validation set.')
    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')
    group.add_argument('--data_path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ... It is used with --split when a '
                       'single dataset used for all three: train, valid '
                       'and test. It is exclusive to the other '
                       '--*-data_path args')
    group.add_argument('--split', type=str, default='969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--train_data_path', nargs='*', default=None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--valid_data_path', nargs='*', default=None,
                       help='Path to the validation dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--test_data_path', nargs='*', default=None,
                       help='Path to the test dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--vocab_file', type=str, default=None,
                       help='Path to the vocab file.')
    group.add_argument('--merge_file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--vocab_extra_ids', type=int, default=0,
                       help='Number of additional vocabulary tokens. '
                            'They are used for span masking in the T5 model')
    group.add_argument('--vocab_extra_ids_list', type=str, default=None,
                       help='comma separated list of special vocab ids to add to the tokenizer')
    group.add_argument('--seq_length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--variable_seq_lengths', action='store_true', default=None,
                       help='Enable variable sequence lengths.')
    group.add_argument('--scalar_loss_mask', type=float, default=0.0,
                       help=('Instruction-tuning argument: Scalar to multiply the '
                             'loss of the "masked out" tokens (usually the user '
                             'tokens, not assistant ones). Set to zero (default) '
                             'to completely remove the loss of said tokens'))
    group.add_argument('--encoder_seq_length', type=int, default=None,
                       help='Maximum encoder sequence length to process.'
                       'This should be exclusive of --seq_length')
    group.add_argument('--decoder_seq_length', type=int, default=None,
                       help="Maximum decoder sequence length to process.")
    group.add_argument('--retriever_seq_length', type=int, default=256,
                       help='Maximum sequence length for the biencoder model '
                       'for retriever')
    group.add_argument('--sample_rate', type=float, default=1.0,
                       help='sample rate for training data. Supposed to be 0 '
                            ' < sample_rate < 1')
    group.add_argument('--mask_prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')
    group.add_argument('--short_seq_prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')
    group.add_argument('--mmap_warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--num_workers', type=int, default=2,
                       help="Dataloader number of workers.")
    group.add_argument('--tokenizer_type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'FalconTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer_model', type=str, default=None,
                       help='Sentencepiece tokenizer model.')
    group.add_argument("--no_new_tokens", action="store_false", dest="new_tokens",
                       help=("Do not add special tokens (e.g. CLS, MASK, etc) "
                             "in the sentenciepiece tokenizer"))
    group.add_argument('--data_impl', type=str, default='infer',
                       choices=['lazy', 'cached', 'mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--reset_position_ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset_attention_mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')
    group.add_argument('--eod_mask_loss', action='store_true',
                       help='Mask loss for the end of document tokens.')
    return parser


def _add_autoresume_args(parser):
    group = parser.add_argument_group(title='autoresume')
    group.add_argument('--adlr_autoresume', action='store_true',
                       help='Enable autoresume on adlr cluster.')
    group.add_argument('--adlr_autoresume_interval', type=int, default=1000,
                       help='Intervals over which check for autoresume'
                       'termination signal')
    return parser


def _add_biencoder_args(parser):
    group = parser.add_argument_group(title='biencoder')
    # network size
    group.add_argument('--ict_head_size', type=int, default=None,
                       help='Size of block embeddings to be used in ICT and '
                        'REALM (paper default: 128)')
    group.add_argument('--biencoder_projection_dim', type=int, default=0,
                       help='Size of projection head used in biencoder')
    group.add_argument('--biencoder_shared_query_context_model', action='store_true',
                        help='Whether to share the parameters of the query '
                        'and context models or not')
    # checkpointing
    group.add_argument('--ict_load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert_load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint '
                       '(needed to start ICT and REALM)')

    # data
    group.add_argument('--titles_data_path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query_in_block_prob', type=float, default=0.1,
                       help='Probability of keeping query in block for '
                       'ICT dataset')
    group.add_argument('--use_one_sent_docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')
    group.add_argument('--evidence_data_path', type=str, default=None,
                       help='Path to Wikipedia Evidence frm DPR paper')

    # training
    group.add_argument('--retriever_report_topk_accuracies', nargs='+', type=int,
                        default=[], help="Which top-k accuracies to report "
                        "(e.g. '1 5 20')")
    group.add_argument('--retriever_score_scaling', action='store_true',
                       help='Whether to scale retriever scores by inverse '
                        'square root of hidden size')

    # faiss index
    group.add_argument('--block_data_path', type=str, default=None,
                       help='Where to save/load BlockData to/from')
    group.add_argument('--embedding_path', type=str, default=None,
                       help='Where to save/load Open-Retrieval Embedding'
                        ' data to/from')

    # indexer
    group.add_argument('--indexer_batch_size', type=int, default=128,
                       help='How large of batches to use when doing indexing '
                       'jobs')
    group.add_argument('--indexer_log_interval', type=int, default=1000,
                       help='After how many batches should the indexer '
                       'report progress')
    return parser


def _add_vision_args(parser):
    group = parser.add_argument_group(title="vision")

    # general vision arguements
    group.add_argument('--num_classes', type=int, default=1000,
                       help='num of classes in vision classificaiton task')
    group.add_argument('--img_h', type=int, default=224,
                       help='Image height for vision classification task')
    group.add_argument('--img_w', type=int, default=224,
                       help='Image height for vision classification task')
    group.add_argument('--num_channels', type=int, default=3,
                       help='Number of channels in input image data')
    group.add_argument('--patch_dim', type=int, default=16,
                       help='patch dimension')
    group.add_argument('--classes_fraction', type=float, default=1.0,
                       help='training with fraction of classes.')
    group.add_argument('--data_per_class_fraction', type=float, default=1.0,
                       help='training with fraction of data per class.')
    group.add_argument('--no_data_sharding', action='store_false',
                       help='Disable data sharding.',
                       dest='data_sharding')
    group.add_argument('--head_lr_mult', type=float, default=1.0,
                       help='learning rate multiplier for head during finetuning')

    # dino arguments
    group.add_argument('--iter_per_epoch', type=int, default=1250,
                       help='iterations per epoch')
    group.add_argument('--dino_local_img_size', type=int, default=96,
                       help='Image size for vision classification task')
    group.add_argument('--dino_local_crops_number', type=int, default=10,
                       help='Number of local crops')
    group.add_argument('--dino_head_hidden_size', type=int, default=2048,
                       help='Hidden dimension size in dino head')
    group.add_argument('--dino_bottleneck_size', type=int, default=256,
                       help='Bottle neck dimension in dino head ')
    group.add_argument('--dino_freeze_last_layer', type=float, default=1,
                       help='Freezing last layer weights')
    group.add_argument('--dino_norm_last_layer', action='store_true',
                       help='Disable Norm in last layer.')
    group.add_argument('--dino_warmup_teacher_temp', type=float, default=0.04,
                       help='warump teacher temperature')
    group.add_argument('--dino_teacher_temp', type=float, default=0.07,
                       help='teacher temperature')
    group.add_argument('--dino_warmup_teacher_temp_epochs', type=int, default=30,
                       help='warmup teacher temperaure epochs')
    return parser
