import math
from typing import Callable

import torch

from megatron import get_tokenizer
from megatron.utils import average_losses_across_data_parallel_group
from megatron.core.tensor_parallel import vocab_parallel_max_indices


class MetricInput:
    def __init__(self, batch: tuple, output: torch.Tensor, loss: torch.Tensor):
        # regular parameters
        (self.tokens, self.labels, self.loss_mask, self.attention_mask,
         self.position_ids) = batch
        self.output = output
        self.loss = loss
        # lazy parameters
        self._max_indices = None
        self._instruct_mask = None

    @property
    def max_indices(self) -> torch.Tensor:
        if self._max_indices is None:
            self._max_indices = vocab_parallel_max_indices(self.output)
        return self._max_indices

    @property
    def instruct_mask(self) -> torch.Tensor:
        if self._instruct_mask is None:
            # like loss_mask but ignoring the <|im_end|> and <|im_start|>role\n too
            tokenizer = get_tokenizer()
            im_start_id, = tokenizer.tokenize("<|im_start|>")
            im_end_id, = tokenizer.tokenize("<|im_end|>")
            should_keep = torch.ones_like(self.loss_mask)
            # mask all indices where <|im_start|> is found plus the next two tokens
            #  (corresponds to the role and the newline)
            i, j = torch.nonzero(self.labels == im_start_id, as_tuple=True)
            if torch.any(j + 2 >= should_keep.size(1)):
                print("Error calculating instruct mask")
                self._instruct_mask = None
                return self._instruct_mask
            should_keep[i, j] = 0.0
            should_keep[i, j + 1] = 0.0
            should_keep[i, j + 2] = 0.0
            # mask <|im_end|> plus the next token (newline) and the next one
            #  that is a weird space or something
            i, j = torch.nonzero(self.labels == im_end_id, as_tuple=True)
            if torch.any(j + 2 >= should_keep.size(1)):
                print("Error calculating instruct mask")
                self._instruct_mask = None
                return self._instruct_mask
            should_keep[i, j] = 0.0
            should_keep[i, j] = 0.0
            should_keep[i, j + 1] = 0.0
            should_keep[i, j + 2] = 0.0
            # update mask
            self._instruct_mask = self.loss_mask*should_keep
        return self._instruct_mask


def perplexity(inputs: MetricInput) -> dict[str, int | float]:
    ppl = math.exp(min(20, inputs.loss.item()))
    return {"ppl": ppl}


def accuracy(inputs: MetricInput) -> dict[str, int | float]:
    matching = torch.masked_fill(inputs.labels == inputs.max_indices,
                                 inputs.loss_mask == 0, False)
    accuracy = torch.count_nonzero(matching)/torch.count_nonzero(inputs.loss_mask)
    averaged_accuracy = average_losses_across_data_parallel_group([accuracy])
    return {"lm accuracy": averaged_accuracy[0]}


# like accuracy but ignoring the <|im_end|> and <|im_start|> in the
# accuracy calculation
def instruct_accuracy(inputs: MetricInput) -> dict[str, int | float]:
    if inputs.instruct_mask is None:
        accuracy = torch.tensor(torch.nan, device=inputs.labels.device)
    else:
        matching = torch.masked_fill(inputs.labels == inputs.max_indices,
                                     inputs.instruct_mask == 0, False)
        accuracy = torch.count_nonzero(matching)/torch.count_nonzero(inputs.instruct_mask)
    averaged_accuracy = average_losses_across_data_parallel_group([accuracy])
    return {"instruct accuracy": averaged_accuracy[0]}


def count_loss_mask(inputs: MetricInput) -> dict[str, int | float]:
    count = torch.count_nonzero(inputs.loss_mask)/inputs.loss_mask.size(0)
    return {"count loss mask": count}


def count_instruct_mask(inputs: MetricInput) -> dict[str, int | float]:
    if inputs.instruct_mask is None:
        return {}
    count = torch.count_nonzero(inputs.instruct_mask)/inputs.instruct_mask.size(0)
    return {"count instruct mask": count}


METRICS = {
    "perplexity": perplexity,
    "accuracy": accuracy,
    "instruct_accuracy": instruct_accuracy,
    "count_loss_mask": count_loss_mask,
    "count_instruct_mask": count_instruct_mask,
}


def get_metric(name: str) -> Callable[[MetricInput], dict[str, int | float]]:
    return METRICS[name]
