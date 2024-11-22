
def main():
    '''
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
    argParser.add_argument("-o","--out_dir",dest="out_dir",action="store",default='default-out',type=str)
    cli_args = argParser.parse_args()

    import torch
    from omegaconf import OmegaConf
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
    from utils import instantiate_from_config, parse_checkpoint_filename
    from utils.lightning.callbacks import Custom_checkpointer, Config_printer
    import pudb
    import json
    import torchvision
    import numpy as np
    from PIL import Image
    from tqdm import tqdm

    # load config
    core_config = OmegaConf.load('configs/core/training_mandatory.yaml') # this script must have this
    base_config = OmegaConf.load(f'configs/{cli_args.config}')
    config = OmegaConf.merge(core_config,base_config)
    os.makedirs(config.instance_data_dir,exist_ok=True)

    # find latest
    checkpoint_root = config.custom_checkpointer.params.checkpoint_root
    checkpoints = os.listdir(checkpoint_root) if os.path.exists(checkpoint_root) else []
    checkpoints = [x for x in checkpoints if x.endswith('.ckpt')]
    checkpoints.sort()
    if len(checkpoints) > 0:
        latest_checkpoint = pjoin(checkpoint_root,checkpoints[-1])
        print(f'Resuming from {latest_checkpoint}')
        ckpt_path = latest_checkpoint
    else:
        print('No checkpoints to load')
        exit()

    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda()

    # find scene id
    scenes_path = pjoin(config.instance_data_dir,cli_args.out_dir,'scenes')
    scenes = os.listdir(scenes_path)
    scenes.sort()

    for cur_scene in tqdm(scenes):
        sampling_instance_root = pjoin(scenes_path,cur_scene)
        sample_root = pjoin(sampling_instance_root,'observed')
        latents_root = pjoin(sample_root,'latents')
        images_root = pjoin(sample_root,'images')
        os.makedirs(latents_root,exist_ok=True)
        ims = os.listdir(images_root)

        with torch.no_grad():
            for im_fn in ims:
                im = Image.open(pjoin(images_root,im_fn))
                im = np.asarray(im).transpose(2,0,1).astype(np.float32)/127.5 - 1
                im = torch.tensor(im)[None,...].cuda()

                encoded_im = model.latent_autoencoder.encode_to_prequant(im).cpu().detach().numpy()
                np.save(pjoin(latents_root,im_fn[:-4]+'.npy'),encoded_im[0,...])

if __name__ == '__main__':
    main()
