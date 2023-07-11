from megatron.arguments import parse_args
from megatron.global_vars import _set_tensorboard_writer,get_tensorboard_writer
if __name__ == "__main__":
    args=parse_args()
    args.wandb_logger="True"
    args.wandb_project="test-logger"
    _set_tensorboard_writer(args)

    writer=get_tensorboard_writer()
