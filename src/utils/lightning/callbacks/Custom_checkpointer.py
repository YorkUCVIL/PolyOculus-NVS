from lightning.pytorch.callbacks import Callback
import time
import os
from os.path import join as pjoin
from lightning.pytorch.utilities import rank_zero_only
from ...parse_checkpoint_filename import parse_checkpoint_filename
import signal

class Custom_checkpointer(Callback):
    '''
    Custom checkpoint callback
     - this callback saves at the BEGINNING of the epoch loop to simplify the trainer state management on resume
     - epochs and steps stored in a checkpoint is the next step that will be run on resume, NOT the last step fully run
     - saves triggered by:
        - signal received
        - epoch end
        - min time between checkpoints
     - other management features:
        - keep n previous checkpoints
        - keep checkpoints at an epoch interval
    '''
    def __init__(self,
                 checkpoint_root,
                 time_interval=-1,
                 keep_n_last=-1,
                 keep_epoch_interval=-1,
                 handle_signals=['SIGTERM','SIGINT'],
                 ):
        super().__init__()
        self._term_sig_recieved = False
        self.last_time_saved = time.time()
        self.checkpoint_root = checkpoint_root
        self.time_interval = time_interval
        self.keep_n_last = keep_n_last
        self.keep_epoch_interval = keep_epoch_interval
        self.block_epoch_save = False

        # configure checkpoint callback
        for sig in handle_signals:
            signal.signal(getattr(signal,sig),self.termination_handler)

    def termination_handler(self,sig_num,frame):
        self._term_sig_recieved = True

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if trainer.global_rank == 0:
            # exit on signal
            if self._term_sig_recieved:
                print(f'Term signal received, checkpointing...')
                trainer.save_checkpoint(pjoin(self.checkpoint_root,f'epoch={trainer.current_epoch:04d}-step={batch_idx:08d}.ckpt'))
                print('Checkpoint complete, exiting.')
                trainer.should_stop = True
                self.block_epoch_save = True
                raise StopIteration
                # we could cull here, but it's best to exist quickly
                return
            if self.time_interval > 0 and time.time() > self.last_time_saved + self.time_interval:
                print(f'Have not checkpointed in over {self.time_interval} seconds, checkpointing...')
                trainer.save_checkpoint(pjoin(self.checkpoint_root,f'epoch={trainer.current_epoch:04d}-step={batch_idx:08d}.ckpt'))
                self.last_time_saved = time.time()
                print('Checkpoint complete')
                self._cull_checkpoints(trainer)
        elif self._term_sig_recieved:
            trainer.should_stop = True
            self.block_epoch_save = True
            time.sleep(5)
            print(f'Term signal received, exiting.')
            raise StopIteration

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        if self.block_epoch_save: return # sig term will trigger this, don't save again
        print('Epoch complete, checkpointing...')
        trainer.save_checkpoint(pjoin(self.checkpoint_root,f'epoch={trainer.current_epoch+1:04d}-step={0:08d}.ckpt'))
        self.last_time_saved = time.time()
        print('Checkpoint complete')
        self._cull_checkpoints(trainer)

    def _cull_checkpoints(self,trainer):
        if self.keep_epoch_interval < 0 and self.keep_n_last < 0: return
        checkpoints = [x for x in os.listdir(self.checkpoint_root) if x.endswith('.ckpt')]
        checkpoints.sort()
        for ckpt in checkpoints:
            epoch, step = parse_checkpoint_filename(ckpt)
            if self.keep_epoch_interval > 0 and epoch%self.keep_epoch_interval==0 and step==0:
                continue
            elif self.keep_n_last > 0 and ckpt in checkpoints[-self.keep_n_last:]:
                continue
            print(f'Clearing checkpoint {ckpt}')
            trainer.strategy.remove_checkpoint(pjoin(self.checkpoint_root,ckpt))
