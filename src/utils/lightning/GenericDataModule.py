import lightning.pytorch as pl
from ..instantiate_from_config import instantiate_from_config
from .Skippable_distributed_sampler import Skippable_distributed_sampler
import torch
import os

class GenericDataModule(pl.LightningDataModule):
    def __init__(self,train=None,validation=None):
        super().__init__()
        self.trainer = None
        self.train_skip_n_examples = 0
        self.train_config = train
        self.val_config = validation
        self.train_dataset = None if self.train_config is None else instantiate_from_config(self.train_config)
        self.val_dataset = None if self.val_config is None else instantiate_from_config(self.val_config)
    def train_dataloader(self):
        if self.train_dataset is None: return None
        assert self.trainer is not None, 'must attach trainer to datamodule before loader creation'
        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        shuffle = self.train_config.loader.shuffle
        batch_size = self.train_config.loader.batch_size
        num_workers = self.train_config.loader.num_workers

        sampler = Skippable_distributed_sampler(self.train_dataset,num_replicas=world_size,rank=rank,shuffle=shuffle,n_skip=self.train_skip_n_examples)
        def worker_init_fn(worker_id): # init function adds debug functionality + seed management, depends on trainer for epoch awareness
            worker_info = torch.utils.data.get_worker_info()
            # set id for tracing
            worker_info.dataset.id = f'{self.trainer.current_epoch}-{self.trainer.global_rank}-{worker_info.seed}'
            worker_info.dataset.init_dataset()
            worker_info.dataset.set_epoch(self.trainer.current_epoch) # change seeds based on epoch
        loader = torch.utils.data.DataLoader(self.train_dataset,batch_size=batch_size,worker_init_fn=worker_init_fn,num_workers=num_workers,sampler=sampler)
        return loader
    def val_dataloader(self):
        if self.val_dataset is None: return None
        assert self.trainer is not None, 'must attach trainer to datamodule before loader creation'
        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        batch_size = self.val_config.loader.batch_size
        num_workers = self.val_config.loader.num_workers
        sampler = Skippable_distributed_sampler(self.val_dataset,num_replicas=world_size,rank=rank,shuffle=False,n_skip=0)
        def worker_init_fn(worker_id): # init function adds debug functionality + seed management, depends on trainer for epoch awareness
            worker_info = torch.utils.data.get_worker_info()
            worker_info.dataset.init_dataset()
            worker_info.dataset.set_epoch(self.trainer.current_epoch) # change seeds based on epoch
        loader = torch.utils.data.DataLoader(self.val_dataset,batch_size=batch_size,worker_init_fn=worker_init_fn,num_workers=num_workers,sampler=sampler)
        return loader
