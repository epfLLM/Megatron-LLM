from megatron.arguments import parse_args
from megatron.global_vars import _set_tensorboard_writer,get_tensorboard_writer
if __name__ == "__main__":
    args=parse_args()
    args.wandb_logger="True"
    args.wandb_project="test-logger"
    _set_tensorboard_writer(args)

    for i,l in zip(range(10),range(10,20)):
        r=10-i
        writer=get_tensorboard_writer()
        for k in range(5):
            writer.add_scalar(f"forward{k}",l,i)
            writer.add_scalar(f"backward{k}",r,i)
            writer.add_scalar(f"forward{k} vs forward",l,i)
            writer.add_scalar(f"forward{k} vs backward",i,r)


