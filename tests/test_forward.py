import os
from datetime import timedelta
import functools

import torch

import megatron
import megatron.initialize
import megatron.tokenizer

from megatron.core import tensor_parallel
import megatron.data
import megatron.data.data_samplers
from megatron import get_tokenizer
from megatron.model import LlamaModel
import megatron.model.transformer
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType, PositionEmbeddingType
import megatron.data.gpt_dataset
from megatron.utils import get_ltor_masks_and_position_ids

from megatron.core import mpu, tensor_parallel


def add_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='validation set')
    # group.add_argument('--micro_batch_size', default=2)
    group.add_argument('--eval_ppl', action='store_true', default=False)
    group.add_argument('--stored_params', type=dict, default=dict())
    # group.add_argument('--padded_vocab_size', type=int, default=100)
    # group.add_argument('--use_flash_attn')

    """
    --max_position_embeddings 13
    --num_layers 2
    --hidden_size 4
    --num_attention_heads 4
    --encoder_seq_length 4
    --global_batch_size 128
    --train_iters 2000000
    --data_impl mmap
    --split 80,10,10
    --distributed_backend nccl
    --lr_decay_style constant
    --lr 0.0001
    --clip_grad 1.0
    --weight_decay 0.1
    --adam_beta1 0.9
    --adam_beta2 0.95
    --eval_iters 50
    --DDP_impl local
    --finetune
    --no_load_optim
    --vocab_file gpt2-vocab.json
    --merge_file gpt2-merges.txt
    --train_iters 2000000
    --data_path /idiap/user/kmatoba/model-parallel-trainer/my_bert_text_sentence
    --data_path2 /idiap/user/kmatoba/model-parallel-trainer/my_bert_text_sentence
    --use_rms_norm
    --glu_activation swiglu
    --use_bias
    --use_post_ln
    """
    return parser


def test_flash_attention1():
    # dtype = torch.float32
    dtype = torch.float16
    q = torch.rand(32, 8, 128, 64, dtype=dtype, device="cuda")
    k = torch.rand(32, 8, 128, 64, dtype=dtype, device="cuda")
    v = torch.rand(32, 8, 128, 64, dtype=dtype, device="cuda")

    y1 = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                     attn_mask=None, dropout_p=0.0, is_causal=True)
    fsa = megatron.model.transformer.FlashSelfAttention()
    y2 = fsa(q, k, v)


def test_parallel_attention_forward():
    init_method_std = 0.02
    num_layers = 2

    init_method = init_method_normal(init_method_std)
    output_layer_init_method = scaled_init_method_normal(init_method_std,
                                                         num_layers)
    layer_number = 1
    world_size = 1

    attention_type = AttnType.self_attn
    attn_mask_type = AttnMaskType.padding

    model_type_llama = ModelType.encoder_or_decoder,
    args_defaults = {'tokenizer_type': 'GPT2BPETokenizer'}
    # args_defaults = {}
    extra_args_provider = lambda _: _

    megatron.initialize.initialize_megatron(extra_args_provider, args_defaults)

    base_parser = megatron.arguments.build_base_parser()
    final_parser = add_args(base_parser)
    args = final_parser.parse_args()
    args.gradient_accumulation_fusion = False

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    n_head = 128
    head_dim = 64
    hidden_size = head_dim * n_head
    split_size = 9
    hidden_dropout = 0.0
    attention_dropout = 0.0
    rotary = True
    n_head_kv = 8
    seq_length = 13
    batch_size = 2
    args.hidden_size = hidden_size

    megatron.arguments.validate_args(args, args_defaults)
    pa = megatron.model.transformer.ParallelAttention(init_method,
                 output_layer_init_method,
                 layer_number,
                 attention_type,
                 attn_mask_type,
                 world_size,
                 args)
    # h: hidden size
    # b: batch size
    # s: sequence length
    # l: number of layers
    # Transformer takes input of size [s, b, h]

    x = torch.randn((batch_size, seq_length, hidden_size), device="cuda")
    attention_mask = None
    output, bias = pa(x, attention_mask)
    print(output, bias)

    # pa.core_attention(query_layer, key_layer, value_layer, attention_mask)


def _model_provider_unwrapped(pre_process: bool,
                              post_process: bool,
                              args):
    # pre_process = False
    parallel_output = True
    num_tokentypes = 0
    model_type_llama = ModelType.encoder_or_decoder
    model = LlamaModel(
        num_tokentypes,
        parallel_output,
        pre_process,
        post_process,
        args,
        model_type_llama
    )
    return model


def _get_batch(data_iterator, args):
    """Generate a batch"""
    tokenizer = megatron.tokenizer.build_tokenizer(args)

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    print(data)
    # data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    data_b = data
    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def test_model_forward():
    tensor_parallel.model_parallel_cuda_manual_seed(seed=12345)

    args_defaults = {'tokenizer_type': 'GPT2BPETokenizer'}
    # args_defaults = {}
    extra_args_provider = lambda _: _
    # megatron.initialize.initialize_megatron(extra_args_provider, args_defaults)

    base_parser = megatron.arguments.build_base_parser()
    final_parser = add_args(base_parser)
    args = final_parser.parse_args()
    args.gradient_accumulation_fusion = False

    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    megatron.arguments.validate_args(args, args_defaults)

    # torch.distributed.init_process_group(
    #     backend=args.distributed_backend,
    #     world_size=args.world_size,
    #     rank=args.rank,
    #     timeout=timedelta(minutes=10)
    # )

    pre_process = True
    post_process = False

    parallel_output = True
    num_tokentypes = 0
    model_type_llama = ModelType.encoder_or_decoder
    model = LlamaModel(
        num_tokentypes,
        parallel_output,
        pre_process,
        post_process,
        args,
        model_type_llama
    )
    data_prefix = ['/idiap/user/kmatoba/model-parallel-trainer/my_bert_text_sentence']
    data_impl = 'mmap'
    splits_string = '80,10,10'
    train_valid_test_num_samples = [256000000, 12806400, 6400]
    seq_length = 4
    seed = 1234
    skip_warmup = True
    train_dataset, valid_dataset, test_dataset = \
        megatron.data.gpt_dataset.build_train_valid_test_datasets(data_prefix,
                                                                data_impl,
                                                                splits_string,
                                                                train_valid_test_num_samples,
                                                                seq_length,
                                                                seed,
                                                                skip_warmup)
    # train_dataloader = megatron.data.data_samplers.build_pretraining_data_loader(train_dataset, args.consumed_train_samples, args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset)

    train_data_iterator = iter(train_dataloader)
    tokens, labels, loss_mask, attention_mask, position_ids = _get_batch(train_data_iterator, args)
    # print(tokens)
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    labels = labels.to(device)
    loss_mask = loss_mask.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)

    megatron.global_vars._build_num_microbatches_calculator(args)
    megatron.global_vars.build_num_microbatches_calculator(args)

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)


if __name__ == "__main__":
    # test_flash_attention1()
    test_model_forward()
