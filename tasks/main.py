# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Main tasks functionality."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
import megatron.initialize


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, required=True,
                       help='Task name.')
    group.add_argument('--epochs', type=int, default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')
    group.add_argument('--pretrained_checkpoint', type=str, default=None,
                       help='Pretrained checkpoint used for finetunning.')
    group.add_argument('--keep_last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')
    group.add_argument('--train_data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')
    group.add_argument('--valid_data', nargs='*', default=None,
                       help='path(s) to the validation data.')
    group.add_argument('--overlapping_eval', type=int, default=32,
                       help='Sliding window for overlapping evaluation.')
    group.add_argument('--strict_lambada', action='store_true',
                       help='Use more difficult formulation of lambada.')
    # Retriever args
    group.add_argument('--qa_data_dev', type=str, default=None,
                       help='Path to the QA dataset dev file.')
    group.add_argument('--qa_data_test', type=str, default=None,
                       help='Path to the QA dataset test file.')

    # Faiss arguments for retriever
    group.add_argument('--faiss_use_gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--faiss_match', type=str, default='string', \
                        choices=['regex', 'string'], help="Answer matching '\
                        'logic type")
    group.add_argument('--faiss_topk_retrievals', type=int, default=100,
                       help='Number of blocks to use as top-k during retrieval')

    # finetune for retriever
    group.add_argument('--eval_micro_batch_size', type=int, default=None,
                       help='Eval Batch size per model instance (local batch '
                            'size). Global batch size is local batch size '
                            'times data parallel size.')
    group.add_argument('--train_with_neg', action='store_true',
                       help='Whether to use negative examples during model training')
    group.add_argument('--train_hard_neg', type=int, default=0,
                       help='Number of hard negative exmaples to use during '
                        'training')

    # parameters for Av.rank validation method
    # Following options/arguments have been taken directly from DPR codebase
    group.add_argument('--val_av_rank_hard_neg', type=int, default=30,
                        help='Av.rank validation: how many hard negatives to'
                        ' take from each question pool')
    group.add_argument('--val_av_rank_other_neg', type=int, default=30,
                        help='Av.rank validation: how many other negatives to'
                        ' take from each question pool')
    return parser


if __name__ == '__main__':
    megatron.initialize.initialize_megatron(extra_args_provider=get_tasks_args)
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for downstream tasks.")
        exit()

    if args.task == 'RACE':
        from race.finetune import main
    elif args.task in ['MNLI', 'QQP']:
        from glue.finetune import main
    elif args.task in ['LAMBADA', 'WIKITEXT103']:
        from zeroshot_gpt.evaluate import main
    elif args.task in ['ICT-ZEROSHOT-NQ', 'RETRIEVER-EVAL']:
        from orqa.evaluate_orqa import main
    elif args.task in ['RET-FINETUNE-NQ']:
        from orqa.supervised.finetune import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.task))

    main()
