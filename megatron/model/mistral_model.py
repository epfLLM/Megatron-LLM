"""Mistral Model."""

import warnings

from megatron import get_args
from .enums import PositionEmbeddingType
from . import GPTModel


class MistralModel(GPTModel):
    def __init__(self,
                 num_tokentypes: int = 0,
                 parallel_output: bool = True,
                 pre_process: bool = True,
                 post_process: bool = True,
                 model_type=None
                 ):

        args = get_args()

        # mandatory arguments
        assert args.position_embedding_type == PositionEmbeddingType.rotary, \
            f"Mistral uses rotary embedding, not {args.position_embedding_type}"
        assert not args.use_post_ln, "Mistral does not use post_ln"
        assert args.glu_activation == "swiglu", "Mistral works with swiglu activation"
        assert not args.use_bias, "Mistral does not use bias"
        assert not args.parallel_attn, "Mistral does not use parallel_attn"
        assert args.use_rms_norm, "Mistral uses rms_norm"
        assert not args.tie_embed_logits , "Mistral unties embedding and lm_head weights"
        assert args.sliding_window == 4096, "Mistral uses sliding window attention (sliding_window=4096)"

        # recomended arguments
        if not args.use_flash_attn:
            warnings.warn("Mistral should use flash attn (for sliding window local attention)")

        if args.bias_gelu_fusion:
            warnings.warn("Mistral is not intended to use bias_gelu_fusion")
        if args.bias_dropout_fusion:
            warnings.warn("Mistral is not intended to use bias_dropout_fusion")
        if args.hidden_dropout > 0.0 and not args.lima_dropout:
            warnings.warn("Mistral is not intended to use dropout")
        if args.attention_dropout > 0.0:
            warnings.warn("Mistral is not intended to use dropout")
        super().__init__(num_tokentypes=num_tokentypes, parallel_output=parallel_output,
                         pre_process=pre_process, post_process=post_process,
                         model_type=model_type)
