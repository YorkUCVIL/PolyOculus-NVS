
def main():
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
    argParser.add_argument("-i","--sampling_index",dest="sampling_index",action="store",default=0,type=int)
    argParser.add_argument("-o","--out_dir",dest="out_dir",action="store",default='default-out',type=str)
    argParser.add_argument("-s","--scene_idx",dest="scene_idx",action="store",default=0,type=int)
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
    cur_scene = scenes[cli_args.scene_idx]

    # load spec
    sampling_instance_root = pjoin(scenes_path,cur_scene)
    spec_path = pjoin(sampling_instance_root,'sampling-spec.json')
    with open(spec_path) as f:
        spec = json.load(f)
        focal = spec['focal_y']
        all_poses = torch.tensor(spec['poses'])
        generation_sets = spec['generation_sets']
        generation_sets_conditioning = spec['generation_sets_conditioning']

    # initialize sampling
    sample_root = pjoin(sampling_instance_root,f'samples/{cli_args.sampling_index:04d}')
    latents_root = pjoin(sample_root,'latents')
    images_root = pjoin(sample_root,'images')
    os.makedirs(latents_root,exist_ok=True)
    os.makedirs(images_root,exist_ok=True)
    init_latents_root = pjoin(sampling_instance_root,'observed','latents')
    init_latents = os.listdir(init_latents_root)
    for lat_fn in init_latents:
        lat_np = np.load(pjoin(init_latents_root,lat_fn))
        lat = torch.tensor(lat_np).cuda()[None,...]
        im = model.latent_autoencoder.decode_code(lat).detach().cpu().numpy()[0,...].transpose(1,2,0)
        im = np.clip((im+1)*127.5,0,255).astype(np.uint8)
        Image.fromarray(im).save(pjoin(images_root,f'{lat_fn[:-4]}.png'))
        np.save(pjoin(latents_root,lat_fn),lat_np)

    # generate all sets sequentially
    assert len(generation_sets) == len(generation_sets_conditioning)
    for gen_set,gen_set_cond in zip(generation_sets,generation_sets_conditioning):
        n_cond = len(gen_set_cond)
        n_views = len(gen_set) + len(gen_set_cond)

        # assemble inputs
        cond_latents = []
        poses = []
        focals = torch.ones(size=[1,n_views]).cuda()*focal
        for cond in gen_set_cond:
            cond_latent = np.load(pjoin(sampling_instance_root,f'samples/{cli_args.sampling_index:04d}','latents',f'{cond:04d}.npy'))
            cond_latent = torch.tensor(cond_latent).cuda()
            cond_latents.append(cond_latent)
            poses.append(all_poses[cond,...].cuda())
        for cond in gen_set:
            poses.append(all_poses[cond,...].cuda())
        poses = torch.stack(poses)[None,...]
        cond_latents = torch.stack(cond_latents)[None,...]

        # do the generation
        input_shape = [1,n_views,3,256,256] # magic!
        with torch.no_grad():
            images, latents = model.p_sample_loop_cond(cond_latents,input_shape,poses,focals, return_latents=True, print_progress=True) # we hardcoded the conditioning batchsize here

            images = torch.clamp((images.cpu().detach()+1)*127.5,0,255)
            images = images[0,...].permute(0,2,3,1).numpy()
            images = images.astype(np.uint8)
            latents = latents.cpu().detach().numpy()[0,...]
            for i, idx in enumerate(gen_set):
                im = images[i+n_cond,...]
                Image.fromarray(im).save(pjoin(images_root,f'{idx:04d}.png'))
                np.save(pjoin(latents_root,f'{idx:04d}.npy'),latents[i+n_cond,...])

if __name__ == '__main__':
    main()
