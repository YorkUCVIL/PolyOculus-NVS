import torch
from torch.utils import data
from os.path import join as pjoin
import os
from PIL import Image
import numpy as np
import json
import sys
from utils import *
import torchvision
import zipfile

class Multi_matterport_dataset(data.Dataset):
	def __init__(self,split,n_views,variable_n_views,internal_dir='webp_keep',format='.webp'):
		super().__init__()
		self.root = f'../dataset-data-matterport/{split}/'
		self.image_root = pjoin(self.root,'images')
		self.poses = np.load(pjoin(self.root,'poses-min-5.npy'),allow_pickle=True).item()
		self.internal_dir = internal_dir
		self.format = format

		self.sequences = []
		for s,eps in self.poses.items():
			for e in eps.keys():
				self.sequences.append((s,e))

		self.n_views = n_views
		self.epoch = 0
		self.variable_n_views = variable_n_views

	def init_dataset(self):
		pass

	def set_epoch(self,epoch):
		self.epoch = epoch

	def __len__(self):
		return len(self.sequences)

	def _set_seeds(self,idx):
		'''
		sets seeds based on epoch and item idx
		should be agnostic to worker number, rank within distributed
		'''
		seed = self.epoch*5000000 + idx
		torch.manual_seed(seed)
		np.random.seed(seed)

	def __getitem__(self,idx):
		self._set_seeds(idx)
		# idx indicates sequence idx, we will pick 2 random frames from there
		scene, episode = self.sequences[idx]
		poses = self.poses[scene][episode]
		available_idxs = list(range(poses.shape[0]))

		# variable number of views
		n_valid_views = np.random.randint(1,self.n_views+1) if self.variable_n_views else self.n_views
		view_mask = np.zeros(shape=[self.n_views]).astype(np.float32)
		view_mask[:n_valid_views] = 1
		view_mask = view_mask == 1 # convert to binary

		# choose frames
		chosen_idxs = np.random.choice(available_idxs,n_valid_views,replace=False)
		
		# get images from zip
		frames = []
		zip_file = zipfile.ZipFile(pjoin(self.image_root,scene,f'{episode:02d}.zip'),'r')
		for im_idx in chosen_idxs:
			with zip_file.open(pjoin(self.internal_dir,scene,f'{episode:02d}',f'{im_idx:04d}{self.format}')) as f:
				frame = Image.open(f).convert('RGB')
				frame = np.asarray(frame)
				frame = frame.transpose(2,0,1)/127.5 - 1
				frames.append(frame)
		zip_file.close()
		ims = np.stack(frames,axis=0)

		# load and process transforms, 3x4 -> 4x4
		poses_out = []
		focals = []
		for im_idx in chosen_idxs:
			tform = poses[im_idx,...]
			poses_out.append(tform)
			focals.append(0.5) # fixed for matterport

		# pad outputs
		required_padding = self.n_views - n_valid_views
		ims = np.concatenate([ims,np.zeros(( required_padding, )+ims.shape[1:]).astype(np.float32)],0).astype(np.float32)
		poses_out = np.concatenate([poses_out]+[np.eye(4)[None,:,:]]*required_padding,0).astype(np.float32)
		focals = np.concatenate([focals,np.ones(( required_padding, )).astype(np.float32)],0).astype(np.float32)

		out_dict = {
			'im': ims[0], # one frame for vqgan training
			'ims': ims,
			'poses': poses_out,
			'focals': focals,
			'view_mask': view_mask
		}
		return out_dict

