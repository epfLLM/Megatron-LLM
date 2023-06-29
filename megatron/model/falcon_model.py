"""Falcon Model."""


from megatron import get_args
from .enums import PositionEmbeddingType
from . import GPTModel


class FalconModel(GPTModel):
    def __init__(self,
                 num_tokentypes: int = 0,
                 parallel_output: bool = True,
                 pre_process: bool = True,
                 post_process: bool = True,
                 model_type=None):
        args = get_args()
        assert args.position_embedding_type == PositionEmbeddingType.rotary, \
            f"Falcon uses rotary embedding, not {args.position_embedding_type}"
        assert args.use_multiquery_attn, "Falcon uses multiquery attn"
        assert isinstance(args.num_attention_heads_kv, int), \
            "Falcon needs a not None num_attention_heads_kv parameter"
        assert not args.use_post_ln, \
                "FalconModel requires pre-normalization, not use_post_ln"
        assert args.glu_activation is None, \
                "FalconModel requires gelu activation (set glu_activation=None)"
        assert not args.use_bias, "Falcon does not use bias"
        assert args.parallel_attn, "Falcon does uses parallel_attn"
        super().__init__(num_tokentypes=num_tokentypes, parallel_output=parallel_output,
                         pre_process=pre_process, post_process=post_process,
                         model_type=model_type)
