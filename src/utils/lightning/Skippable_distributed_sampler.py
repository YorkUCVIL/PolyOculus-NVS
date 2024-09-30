import torch
import math

class Skippable_distributed_sampler(torch.utils.data.distributed.DistributedSampler):
    '''
    An modified distributed sampler
     - allows some idxs to be skipped for the first epoch
     - allows world configuration to be updated after creation
    '''
    def __init__(self,*args,n_skip=0,**kwargs):
        super().__init__(*args,**kwargs)
        self.n_skip = n_skip
        self.initial_epoch_set = False
    def update_world_config(self,num_replicas,rank):
        self.rank = rank
        self.num_replicas = num_replicas
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.n_skip:]
        assert len(indices) == self.num_samples - self.n_skip

        return iter(indices)
    def set_epoch(self, epoch: int) -> None:
        if self.initial_epoch_set: self.n_skip = 0 # turn off skipping after first epoch run, hack
        self.initial_epoch_set = True
        self.epoch = epoch
