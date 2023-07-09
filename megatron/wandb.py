from dataclasses import dataclass,field
from typing import Optional
import torch
from typing import Union
import wandb
from argparse import Namespace
from collections import defaultdict
import os

#TODO: do we want to use the watch? https://docs.wandb.ai/ref/python/watch
@dataclass
class WandBConfig(object):
    # compatibility to previous logger
    # config to be logged in wandb
    config:Optional[Namespace]=field(default_factory=None)
    # NOTE: these are ignored by wandb
    # filled from `tensorboard_log_interval`, frequency of logging
    log_interval:int=field(default=1)
    # filled from `tensorboard_queue_size`, ignored by wandb
    queue_size:int=field(default=int(1e3))
    # if set, log locally into this dir, gets filled from tensorboard_dir
    local_dir:Option[str]=field(default=None)

    # wandb specific config
    # filled from args
    #filled with kwargs
    entity:str=field(default="meditron")
    #wandb project to log to
    project:str=field(default="meditron")
    # save the code to the notebook?
    save_code:bool=field(default=False)
    # stuff to filter by
    tags:Optional[List[str]]=field(default=None)
    # short descriptive name to quickly find things
    name:Optional[str]=field(default=None)
    # long form notes and info to store
    notes:Optional[str]=field(default=None)
    # TODO: discuss how we want to do this, do we want to resume logging?
    resume:str=field(default="must")
    # TODO: can fill this from the environment variable `WANDB_RUN_ID` ?
    # globally unique id, if passed
    run_id:Optional[str]=field(default=None)
    # "magic" auto instrumentation, see https://docs.wandb.ai/ref/python/init
    magic:bool=field(default=False)
    api_key:Optional[str]=field(default=None)

    @staticmethod
    def from_args(args)->'WandBConfig':
        assert args.rank==args.world_size-1, f"Only supposed to launch on rank {args.rank+1}, but got {args.rank}"
        # following the megatron setup for now, could also do groups instead: https://docs.wandb.ai/guides/track/log/distributed-training
        return WandBConfig(local_dir=args.local_dir,
                           queue_size=args.queue_size,
                           log_interval=args.log_interval,
                           config=args,entity=args.wandb_entity,
                           project=args.wandb_project,
                           run_id=args.wandb_id,
                           resumse=args.wandb_resume,
                           api_key=wargs.wandb_api_key
                           )

class WandbTBShim(object):
    """
    Shim class that holds the configuration we want the wandb wrapper to use
    (e.g. to control sampling, delay upload etc) and that translates the API
    """
    def __init__(self, config:WandBConfig):
        super().__init__()
        self.cfg=config
        if os.environ.get("WANDB_API_KEY") is None:
            if self.cfg.api_key is None:
                raise ValueError("WANDB_API_KEY is not set, nor passed as an argument")
            else:
                os.environ["WANDB_API_KEY"]=self.cfg.api_key
        wandb.init(config=config.config,
                   entity=config.entity,
                   project=config.project,
                   save_code=config.save_code,
                   tags=config.tags,
                   name=config.name,
                   notes=config.notes,
                   resume=config.resume,
                   id=config.run_id
                )
        self._last_step_vs:{None:None}
        self._log_accum=defaultdict(dict)
    def add_scalar(self,name:str,var:Union[float,int,torch.Tensor],step:int):
        vs= None if " vs " not in name else name.split(" vs ")[-1]
        if self._last_step[vs] is not None and step!=self._last_step[vs]:
            self._flush(vs)
        self._last_step[vs]=step
        self._log_accum[vs][name]=var
    def _flush(self,vs:Optional[str]):
        if self._log_accum[vs]:
            wandb.log(self._log_accum[vs],step=self._last_step[vs],commit=True)
        self._log_accum[vs]={}
        self._last_step[vs]=None
    def flush_all(self):
        for k in list(self._log_accum.keys()):
            self._flush(k)

    def add_text(self,name:str,value:str):
        # we log this on the creation of the wandb object, hence no need to log it here
        pass
