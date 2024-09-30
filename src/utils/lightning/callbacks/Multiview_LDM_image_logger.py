import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
import torch
import torchvision
import numpy as np
import os
from PIL import Image
import wandb
import pudb

class Multiview_LDM_image_logger(Callback):
    def __init__(self, batch_frequency, max_images, increase_log_steps):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]

    @rank_zero_only
    def _wandb(self, pl_module, images, global_step, split):
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)

        grids['trainer/global_step'] = global_step
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if self.check_frequency(batch_idx) and self.max_images > 0:
            is_train = pl_module.training # remember train state
            if is_train: pl_module.eval()

            with torch.no_grad():
                # sample_shape = [self.max_images]+list(batch['ims'].shape[1:])
                sample_shape = [1]+list(batch['ims'].shape[1:])
                images = pl_module.p_sample_loop(sample_shape,batch['poses'][0:1,...],batch['focals'][0:1,...]) # we hardcoded the conditioning batchsize here
                images = {
                    'sampled_image':images[0,...], # max_images basically gets ignored
                    'gt_image':batch['ims'][0,...],
                }

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    images[k] = torch.clamp(images[k], -1., 1.)

            # global step is redefined as of lightning 1.6
            actual_global_step = pl_module.trainer.fit_loop.epoch_loop.total_batch_idx
            self.log_local(pl_module.logger.save_dir, split, images, actual_global_step, pl_module.current_epoch, batch_idx)

            logger_class = type(pl_module.logger)
            logger_log_images = self.logger_log_images.get(logger_class, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, actual_global_step, split)

            if is_train: pl_module.train() # restore train state

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")
