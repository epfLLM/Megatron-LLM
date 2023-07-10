# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Inference API."""


import torch

from megatron import get_args
from megatron.core import mpu
from .communication import broadcast_float_list
import megatron.text_generation.generation
from .tokenization import (
    tokenize_prompts,
    detokenize_generations)


def generate_and_post_process(model,
                              prompts=None,
                              tokens_to_generate=0,
                              return_output_log_probs=False,
                              top_k_sampling=0,
                              top_p_sampling=0.0,
                              top_p_decay=0.0,
                              top_p_bound=0.0,
                              temperature=1.0,
                              add_BOS=False,
                              use_eod_token_for_early_termination=True,
                              stop_on_double_eol=False,
                              stop_on_eol=False,
                              prevent_newline_after_colon=False,
                              random_seed=-1,
                              args=None):
    """Run inference and post-process outputs, i.e., detokenize,
    move to cpu and convert to list."""
    assert args is not None

    # Main inference.
    tokens, lengths, output_log_probs = _generate(
        model,
        prompts=prompts,
        tokens_to_generate=tokens_to_generate,
        return_output_log_probs=return_output_log_probs,
        top_k_sampling=top_k_sampling,
        top_p_sampling=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        add_BOS=add_BOS,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        random_seed=random_seed,
        args=args)

    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        tokens, prompts_plus_generations, prompts_plus_generations_segments = \
            detokenize_generations(tokens, lengths, True)

        if return_output_log_probs:
            output_log_probs = output_log_probs.cpu().numpy().tolist()
            for i, (prob, seg) in enumerate(zip(output_log_probs, prompts_plus_generations_segments)):
                output_log_probs[i] = prob[:len(seg)-1]

        return prompts_plus_generations, prompts_plus_generations_segments, \
            output_log_probs, tokens
    return None


def _generate(model,
              prompts=None,
              tokens_to_generate=0,
              return_output_log_probs=False,
              top_k_sampling=0,
              top_p_sampling=0.0,
              top_p_decay=0.0,
              top_p_bound=0.0,
              temperature=1.0,
              add_BOS=False,
              use_eod_token_for_early_termination=True,
              stop_on_double_eol=False,
              stop_on_eol=False,
              prevent_newline_after_colon=False,
              random_seed=-1,
              args=None):
    """Given prompts and input parameters, run inference and return:
       tokens: prompts plus the generated tokens.
       lengths: length of the prompt + generations. Note that we can
           discard tokens in the tokens tensor that are after the
           corresponding length.
       output_log_probs: log probs of the tokens.
    """
    assert args is not None
    padded_vocab_size = args.padded_vocab_size
    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate,
              return_output_log_probs,
              top_k_sampling, top_p_sampling, top_p_decay, top_p_bound,
              temperature, add_BOS, use_eod_token_for_early_termination,
              stop_on_double_eol,
              stop_on_eol,
              prevent_newline_after_colon,
              random_seed]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    return_output_log_probs = bool(values_float_tensor[1].item())
    top_k_sampling = int(values_float_tensor[2].item())
    top_p_sampling = values_float_tensor[3].item()
    top_p_decay = values_float_tensor[4].item()
    top_p_bound = values_float_tensor[5].item()
    temperature = values_float_tensor[6].item()
    add_BOS = bool(values_float_tensor[7].item())
    use_eod_token_for_early_termination = bool(values_float_tensor[8].item())
    stop_on_double_eol = bool(values_float_tensor[9].item())
    stop_on_eol = bool(values_float_tensor[10].item())
    prevent_newline_after_colon = bool(values_float_tensor[11].item())
    random_seed = int(values_float_tensor[12].item())

    if random_seed != -1:
        torch.random.manual_seed(random_seed)

    # Tokenize prompts and get the batch.
    # Note that these tensors are broadcasted to all ranks.
    if torch.distributed.get_rank() == 0:
        assert prompts is not None
    
    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)

    if tokens_to_generate == 0:
        return megatron.text_generation.generation.score_and_return_on_first_stage(
            model, context_tokens_tensor, context_length_tensor, padded_vocab_size, args)

    # Main inference function.
    # Note that the outputs are available on the first stage.
    return megatron.text_generation.generation.generate_tokens_probs_and_return_on_first_stage(
        model, context_tokens_tensor, context_length_tensor,
        return_output_log_probs=return_output_log_probs,
        top_k=top_k_sampling,
        top_p=top_p_sampling,
        top_p_decay=top_p_decay,
        top_p_bound=top_p_bound,
        temperature=temperature,
        use_eod_token_for_early_termination=use_eod_token_for_early_termination,
        stop_on_double_eol=stop_on_double_eol,
        stop_on_eol=stop_on_eol,
        prevent_newline_after_colon=prevent_newline_after_colon,
        padded_vocab_size=padded_vocab_size,
        args=args)


def beam_search_and_post_process(model,
                                 prompts=None,
                                 tokens_to_generate=0,
                                 beam_size=0,
                                 add_BOS=False,
                                 stop_token=50256,
                                 num_return_gen=1,
                                 length_penalty=1,
                                 prevent_newline_after_colon=False,
                                 padded_vocab_size: int=None,
                                 args=None):
    """Run beam search and post-process outputs, i.e., detokenize,
    move to cpu and convert to list."""

    # Main inference.
    tokens, scores = _beam_search(model,
                                  prompts=prompts,
                                  tokens_to_generate=tokens_to_generate,
                                  beam_size=beam_size,
                                  add_BOS=add_BOS,
                                  stop_token=stop_token,
                                  num_return_gen=num_return_gen,
                                  length_penalty=length_penalty,
                                  prevent_newline_after_colon=prevent_newline_after_colon,
                                  padded_vocab_size=padded_vocab_size,
                                  args=args)
    # Only post-process on first stage.
    if mpu.is_pipeline_first_stage():
        lengths = tokens.size(1)*torch.ones(beam_size, dtype=torch.int64, device=torch.cuda.current_device()) 
        tokens, prompts_plus_generations, prompts_plus_generations_segments = detokenize_generations(tokens, lengths, True)
        scores = scores.cpu().numpy().tolist()
        return prompts_plus_generations, prompts_plus_generations_segments, scores

    return None


def _beam_search(model,
                 prompts=None,
                 tokens_to_generate=0,
                 beam_size=0,
                 add_BOS=False,
                 stop_token=50256,
                 num_return_gen=1,
                 length_penalty=1,
                 prevent_newline_after_colon=False,
                 padded_vocab_size=None,
                 args=None):
    assert padded_vocab_size is not None
    assert args is not None
    # Make sure input params are avaialble to all ranks.
    values = [tokens_to_generate,
              beam_size,
              add_BOS,
              stop_token,
              num_return_gen,
              length_penalty,
              prevent_newline_after_colon]
    values_float_tensor = broadcast_float_list(len(values), float_list=values)
    tokens_to_generate = int(values_float_tensor[0].item())
    beam_size = int(values_float_tensor[1].item())
    add_BOS = bool(values_float_tensor[2].item())
    stop_token = int(values_float_tensor[3].item())
    num_return_gen = int(values_float_tensor[4].item())
    length_penalty = values_float_tensor[5].item()
    prevent_newline_after_colon = values_float_tensor[6].item()

    context_tokens_tensor, context_length_tensor = tokenize_prompts(
        prompts=prompts, tokens_to_generate=tokens_to_generate, add_BOS=add_BOS)
    
    return megatron.text_generation.generation.beam_search_and_return_on_first_stage(model, context_tokens_tensor, context_length_tensor,
            beam_size, stop_token=stop_token, num_return_gen=num_return_gen, length_penalty=length_penalty,
            prevent_newline_after_colon=prevent_newline_after_colon, padded_vocab_size=padded_vocab_size, args=args)
