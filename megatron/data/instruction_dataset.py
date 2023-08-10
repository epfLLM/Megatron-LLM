import time
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from megatron.utils import print_rank_0
import megatron.data.indexed_dataset
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import (
    get_train_valid_test_split_,
    get_datasets_weights_and_num_samples,
)


class InstructionDataset(Dataset):
    def __init__(
        self,
        name: str,
        sample_indices: np.ndarray,
        indexed_datasets: dict[str, Dataset],
        seq_length: int,
    ):
        self.indexed_text = indexed_datasets["text"]
        self.indexed_role = indexed_datasets["role"]

        # validate indices
        assert np.min(sample_indices) >= 0
        assert np.max(sample_indices) < len(self.indexed_text)
        assert len(self.indexed_text) == len(self.indexed_role)

        self.name = name
        self.sample_indices = sample_indices
        self.seq_length = seq_length

    def __len__(self) -> int:
        return self.sample_indices.shape[0]

    def __getitem__(self, idx) -> dict:
        # Get the shuffled index.
        idx = self.sample_indices[idx]
        text = self.indexed_text.get(idx)
        role = self.indexed_role.get(idx)
        assert text is not None and role is not None and text.shape == role.shape
        return {
            "text": text.astype(np.int64),
            "role": role.astype(np.int64),
        }


def get_indexed_datasets_(
    data_prefix, data_impl: str, skip_warmup: bool
) -> dict[str, Dataset]:
    print_rank_0(" > building dataset index ...")

    start_time = time.time()
    indexed_text = megatron.data.indexed_dataset.make_dataset(
        data_prefix + "-text",
        data_impl,
        skip_warmup,
    )
    indexed_role = megatron.data.indexed_dataset.make_dataset(
        data_prefix + "-role",
        data_impl,
        skip_warmup,
    )
    assert indexed_text is not None
    print_rank_0(
        " > finished creating indexed dataset in {:4f} seconds".format(
            time.time() - start_time
        )
    )
    num_docs = len(indexed_text)
    print_rank_0("    number of documents: {}".format(num_docs))

    indices = np.arange(
        start=0,
        stop=num_docs,
        step=1,
        dtype=np.int32,
    )
    n_tokens = np.sum(indexed_text.sizes[indices])

    print_rank_0("    number of tokens: {}".format(n_tokens))
    return {"text": indexed_text, "role": indexed_role}


def _sample_dataset(
    np_rng: np.random.RandomState,
    document_indices: np.ndarray,
    indexed_datasets: dict[str, Dataset],
    name: str,
    num_samples: int,
    seq_length: int,
) -> InstructionDataset | None:
    """Compute randomized index of samples for all epochs (num_samples)"""
    assert num_samples > 0

    remaining = num_samples
    index_list = []
    while remaining > 0:
        count = min(remaining, len(document_indices))
        index_list.append(np_rng.permutation(document_indices)[:count])
        remaining -= count
    sample_indices = np.concatenate(index_list)

    dataset = InstructionDataset(
        name,
        sample_indices,
        indexed_datasets,
        seq_length,
    )
    return dataset


def _build_dataset_kernel(
    dataset_name: str,
    data_prefix,
    data_impl: str,
    num_samples: int,
    seq_length: int,
    seed: int,
    skip_warmup: bool,
) -> InstructionDataset:
    """
    Build dataset. This method is called when individual
    train, valid, test datasets are provided
    """

    # Indexed dataset.
    indexed_datasets = get_indexed_datasets_(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = len(indexed_datasets["text"])

    print_rank_0("    {}:".format(dataset_name))
    print_rank_0(
        "     document indices in [0, {}) total of {} "
        "documents".format(total_num_of_documents, total_num_of_documents)
    )

    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
    np_rng = np.random.RandomState(seed=seed)
    dataset = _sample_dataset(
        np_rng, documents, indexed_datasets, dataset_name, num_samples, seq_length
    )

    return dataset


def _build_dataset(
    dataset_name: str,
    data_prefix,
    data_impl: str,
    num_samples: int,
    seq_length: int,
    seed: int,
    skip_warmup: bool,
):
    dataset = None
    if len(data_prefix) == 1:
        dataset = _build_dataset_kernel(
            dataset_name,
            data_prefix[0],
            data_impl,
            num_samples,
            seq_length,
            seed,
            skip_warmup,
        )
    else:
        # Blending dataset.
        # Parse the values.
        output = get_datasets_weights_and_num_samples(data_prefix, num_samples)
        prefixes, weights, dataset_num_samples = output

        # Build individual datasets.
        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset_kernel(
                dataset_name,
                prefixes[i],
                data_impl,
                dataset_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
            )
            if ds:
                datasets.append(ds)

        if datasets:
            dataset = BlendableDataset(datasets, weights)
    return dataset


def _build_train_valid_test_datasets(
    data_prefix,
    data_impl: str,
    splits_string: str,
    train_valid_test_num_samples: list[int],
    seq_length: int,
    seed: int,
    skip_warmup: bool,
):
    """Build train, valid, and test datasets."""

    # Indexed dataset.
    indexed_datasets = get_indexed_datasets_(data_prefix, data_impl, skip_warmup)

    total_num_of_documents = len(indexed_datasets["text"])
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(" > dataset split:")

    def print_split_stats(name, index):
        print_rank_0("    {}:".format(name))
        print_rank_0(
            "     document indices in [{}, {}) total of {} "
            "documents".format(
                splits[index], splits[index + 1], splits[index + 1] - splits[index]
            )
        )

    print_split_stats("train", 0)
    print_split_stats("validation", 1)
    print_split_stats("test", 2)

    # generate random permutation of documents
    np_rng = np.random.RandomState(seed=seed)
    document_indices = np_rng.permutation(total_num_of_documents)

    def _create_dataset(index: int, name: str):
        begin, end = splits[index], splits[index + 1]
        if end <= begin:
            return None

        split_subset = document_indices[begin:end]
        num_samples = train_valid_test_num_samples[index]
        return _sample_dataset(
            np_rng,
            split_subset,
            indexed_datasets,
            name,
            num_samples,
            seq_length,
        )

    train_dataset = _create_dataset(0, "train")
    valid_dataset = _create_dataset(1, "valid")
    test_dataset = _create_dataset(2, "test")

    return train_dataset, valid_dataset, test_dataset


def build_train_valid_test_datasets(
    data_prefix: Optional[str],
    data_impl: str,
    splits_string: str,
    train_valid_test_num_samples: list[int],
    seq_length: int,
    seed: int,
    skip_warmup: bool,
    train_data_prefix=None,
    valid_data_prefix=None,
    test_data_prefix=None,
):
    """Build train, valid, and test datasets."""
    if data_prefix:
        print_rank_0("Single data path provided for train, valid & test")
        # Single dataset.
        if len(data_prefix) == 1:
            return _build_train_valid_test_datasets(
                data_prefix[0],
                data_impl,
                splits_string,
                train_valid_test_num_samples,
                seq_length,
                seed,
                skip_warmup,
            )
        # Blending dataset.
        # Parse the values.
        (
            prefixes,
            weights,
            datasets_train_valid_test_num_samples,
        ) = get_datasets_weights_and_num_samples(
            data_prefix, train_valid_test_num_samples
        )

        # Build individual datasets.
        train_datasets = []
        valid_datasets = []
        test_datasets = []
        for i in range(len(prefixes)):
            train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
                prefixes[i],
                data_impl,
                splits_string,
                datasets_train_valid_test_num_samples[i],
                seq_length,
                seed,
                skip_warmup,
            )
            if train_ds:
                train_datasets.append(train_ds)
            if valid_ds:
                valid_datasets.append(valid_ds)
            if test_ds:
                test_datasets.append(test_ds)

        # Blend.
        blending_train_dataset = None
        if train_datasets:
            blending_train_dataset = BlendableDataset(train_datasets, weights)
        blending_valid_dataset = None
        if valid_datasets:
            blending_valid_dataset = BlendableDataset(valid_datasets, weights)
        blending_test_dataset = None
        if test_datasets:
            blending_test_dataset = BlendableDataset(test_datasets, weights)

        return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)
    else:
        print_rank_0(
            "Separate data paths provided for train, valid & test. Split string will be ignored."
        )
        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        if train_data_prefix is not None:
            train_dataset = _build_dataset(
                "train",
                train_data_prefix,
                data_impl,
                train_valid_test_num_samples[0],
                seq_length,
                seed,
                skip_warmup,
            )

        if valid_data_prefix is not None:
            valid_dataset = _build_dataset(
                "valid",
                valid_data_prefix,
                data_impl,
                train_valid_test_num_samples[1],
                seq_length,
                seed,
                False,
            )

        if test_data_prefix is not None:
            test_dataset = _build_dataset(
                "test",
                test_data_prefix,
                data_impl,
                train_valid_test_num_samples[2],
                seq_length,
                seed,
                False,
            )
        return train_dataset, valid_dataset, test_dataset
