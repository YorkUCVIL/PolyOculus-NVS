import torch
from torch.utils import data
from os.path import join as pjoin
import os
from PIL import Image
import numpy as np
import json
from utils import *
import zipfile
import torchvision
from .sequence_blacklist import sequence_blacklist
torchvision.set_video_backend('video_reader')

class Multi_realestate_dataset(data.Dataset):
	def __init__(self,split,n_views,variable_n_views):
		super().__init__()
		self.dataset_root = f'../dataset-data-realestate/data/{split}'
		self.video_root = pjoin(self.dataset_root,'videos')
		self.poses = np.load(pjoin(self.dataset_root,'poses.npy'),allow_pickle=True).item()

		# remove bad seequences
		for code in sequence_blacklist[split]:
			del self.poses[code]

		self.sequence_codes = list(self.poses.keys())
		self.n_views = n_views
		self.epoch = 0
		self.variable_n_views = variable_n_views

	def init_dataset(self):
		pass

	def set_epoch(self,epoch):
		self.epoch = epoch

	def __len__(self):
		return len(self.sequence_codes)

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
		# idx indicates sequence idx, we will pick many random frames from there
		seq_code = self.sequence_codes[idx]
		poses = self.poses[seq_code]
		sequence_max_len = len(poses['timestamp'])
		assert sequence_max_len >= self.n_views, f'sequence {seq_code} does not have enough views!'
		available_idxs = list(range(len(poses['timestamp'])-1)) # exclude last frame, might have fade between shots

		# variable number of views
		n_valid_views = np.random.randint(1,self.n_views+1) if self.variable_n_views else self.n_views
		view_mask = np.zeros(shape=[self.n_views]).astype(np.float32)
		view_mask[:n_valid_views] = 1
		view_mask = view_mask == 1 # convert to binary

		# choose frames
		chosen_idxs = np.random.choice(available_idxs,n_valid_views,replace=False)

		# extract frames from video
		video_code = poses['video_code']
		video_path = pjoin(self.video_root,f'{video_code}.mp4')
		video_reader = torchvision.io.VideoReader(video_path)
		frames = []
		for im_idx in chosen_idxs:
			timestamp = poses['timestamp'][im_idx]
			seconds = timestamp/1000000
			video_reader.seek(seconds)
			frame = next(video_reader)['data'].numpy().transpose(1,2,0)

			# ensure images have 360 height
			new_w = round(frame.shape[1] * (360/frame.shape[0]))
			frame = np.asarray(Image.fromarray(frame).resize((new_w,360)))
			frames.append(frame)

		# crop and downsample
		cropped_frames = []
		for frame in frames:
			left_pos = (frame.shape[1]-360)//2
			frame_cropped = frame[:,left_pos:left_pos+360,:]
			im = Image.fromarray(frame_cropped).resize((256,256))
			im = np.asarray(im).transpose(2,0,1)/127.5 - 1
			cropped_frames.append(im)
		ims = np.stack(cropped_frames,axis=0)

		# extract poses and focals
		poses_out = []
		focals = []
		for im_idx in chosen_idxs:
			tform = np.concatenate([poses['pose'][im_idx],[[0,0,0,1]]],0)
			focal = poses['focal_y'][im_idx] # these should be the same
			poses_out.append(tform)
			focals.append(focal)

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
