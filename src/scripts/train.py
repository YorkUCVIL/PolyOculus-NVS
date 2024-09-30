
def main():
    '''
    This script glues all the things we need for distributed training
     - loading configs
     - resuming from checkpoints
     - custom checkpointing behavior
     - logging with wandb
     - setting up the datamodule with extra stuff for mid epoch resume
     - printing config to stdout on startup
     - attaching custom callbacks
     - building and running the trainer
    '''
    import os
    import sys
    from os.path import join as pjoin
    import argparse

    # make this script agnostic to cwd
    script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    src_path = os.path.abspath(os.path.join(script_path,'..'))
    sys.path.append(src_path)
    os.chdir(src_path)
    sys.argv[0] = 'scripts/'+sys.argv[0].split('/')[-1] # fix altered path for lightning

    argParser = argparse.ArgumentParser(description='')
    argParser.add_argument("-c","--config",dest="config",action="store",default='',type=str)
    argParser.add_argument("--num_nodes",dest="num_nodes",action="store",default=1,type=int)
    argParser.add_argument("--num_devices",dest="num_devices",action="store",default='auto',type=str)
    cli_args = argParser.parse_args()

    import torch
    from omegaconf import OmegaConf
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
    from utils import instantiate_from_config, parse_checkpoint_filename
    from utils.lightning.callbacks import Custom_checkpointer, Config_printer
    import pudb

    # load config
    core_config = OmegaConf.load('configs/core/training_mandatory.yaml') # this script must have this
    base_config = OmegaConf.load(f'configs/{cli_args.config}')
    config = OmegaConf.merge(core_config,base_config)
    os.makedirs(config.instance_data_dir,exist_ok=True)

    # Check for resume
    checkpoint_root = config.custom_checkpointer.params.checkpoint_root
    checkpoints = os.listdir(checkpoint_root) if os.path.exists(checkpoint_root) else []
    checkpoints = [x for x in checkpoints if x.endswith('.ckpt')]
    checkpoints.sort()
    ckpt_path = None
    resume_step = 0
    if len(checkpoints) > 0:
        latest_checkpoint = pjoin(checkpoint_root,checkpoints[-1])
        print(f'Resuming from {latest_checkpoint}')
        ckpt_path = latest_checkpoint
        _, resume_step = parse_checkpoint_filename(latest_checkpoint)

    # check existing wandb
    logger_kwargs = {}
    if config.get('logger') is not None and config.logger.python_class == 'lightning.pytorch.loggers.WandbLogger':
        wandb_root = pjoin(config.logger.params.save_dir,'wandb')
        if os.path.exists(pjoin(wandb_root)):
            runs = [x for x in os.listdir(wandb_root) if x.startswith('run-')]
            runs.sort()
            if len(runs) > 0:
                logger_kwargs = {'id':runs[-1].split('-')[-1]}

    # create trainer
    config_printer = Config_printer(config) # harded coded, requires access to config
    checkpointer_callback = instantiate_from_config(config.custom_checkpointer)
    trainer_kwargs = OmegaConf.to_container(config.trainer)
    trainer_kwargs['devices'] = cli_args.num_devices if cli_args.num_devices == 'auto' else int(cli_args.num_devices)
    trainer_kwargs['num_nodes'] = cli_args.num_nodes
    trainer_kwargs['callbacks'] = [config_printer,checkpointer_callback]
    if config.trainer.get('callbacks') is not None:
        for cb_spec in config.trainer.callbacks:
            trainer_kwargs['callbacks'].append(instantiate_from_config(cb_spec))
    trainer_kwargs['logger'] = None if config.get('logger') is None else instantiate_from_config(config.logger,logger_kwargs)
    trainer = pl.Trainer(**trainer_kwargs)

    # Create data, and set n samples skipped due to resume
    datamodule = instantiate_from_config(config.data)
    datamodule.train_skip_n_examples = resume_step * config.data.params.train.loader.batch_size

    # create model and train
    model = instantiate_from_config(config.model)
    trainer.fit(model=model, datamodule=datamodule,ckpt_path=ckpt_path)

if __name__ == '__main__':
    main()
