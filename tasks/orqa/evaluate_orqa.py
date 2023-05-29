# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Main tasks functionality."""

from megatron import get_args, print_rank_0
import megatron.indexer
import tasks.orqa.evaluate_utils


def main():
    """
    Main program
    """

    args = get_args()

    """
    Create a BlockData data structure by running an IndexBuilder over an
    ICT Dataset and then evaluate on NQ task
    """

    print_rank_0("Starting index builder!")

    index_builder = megatron.indexer.IndexBuilder(args)
    index_builder.build_and_save_index()
    print_rank_0("Build and save indices: done!")
    print_rank_0("Starting evaluations!")

    # Set up the model and evaluator
    evaluator = tasks.orqa.evaluate_utils.ORQAEvaluator()

    # Run evaluation
    if args.qa_data_dev is not None:
        evaluator.evaluate(args.qa_data_dev, "DEV")

    if args.qa_data_test is not None:
        evaluator.evaluate(args.qa_data_test, "TEST")

