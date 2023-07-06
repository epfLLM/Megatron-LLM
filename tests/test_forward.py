import os

import torch

import megatron
import megatron.initialize
import megatron.model.transformer
from megatron.model.utils import init_method_normal, scaled_init_method_normal
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType, PositionEmbeddingType


def add_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='validation set')
    # group.add_argument('--micro_batch_size', default=2)
    group.add_argument('--eval_ppl', action='store_true', default=False)
    group.add_argument('--stored_params', type=dict, default=dict())
    group.add_argument('--padded_vocab_size', type=int, default=100)
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


if __name__ == "__main__":
    # test_flash_attention1()

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

    hidden_size = 64 * 128
    n_head = 128
    head_dim = 64
    split_size = 9
    hidden_dropout = 0.0
    attention_dropout = 0.0
    rotary = True
    n_head_kv = 8
    seq_length = 13
    batch_size = 2

    megatron.arguments.validate_args(args, args_defaults)
    pa = megatron.model.transformer.ParallelAttention(init_method,
                 output_layer_init_method,
                 layer_number,
                 attention_type,
                 attn_mask_type,
                 world_size,
                 args)
    x = torch.randn((batch_size, seq_length, 4), device="cuda")
    attention_mask = None
    output, bias = pa(x, attention_mask)
    print(output, bias)

    pa.core_attention(query_layer, key_layer, value_layer, attention_mask)



