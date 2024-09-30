
import torch.nn as nn
import torch.nn.functional as F
import torch

class Ray_encoder(nn.Module):
    def __init__(self,resolutions,n_frequencies,half_period):
        super().__init__()
        self.resolutions = resolutions
        self.n_frequencies = n_frequencies
        self.half_period = half_period
        self.n_out_channels = 6 + 2*6*self.n_frequencies

    def to_rays(self,poses,focals,resolution):
        """
        Creates a ray representation based on camera parameters at a specific resolution.
        :param poses: poses as 4x4 matricies [*, 4, 4]
        :param focals: focal lengths [*]
        :param resolution: resolution, assuming image is square
        :type resolution: int
        :return: an [* x 6 x H x W] tensor of rays, where H and W are specified by the resolution.
        """
        extra_dims = poses.shape[:-2]
        n_extra_dims = len(extra_dims)
        assert n_extra_dims == focals.dim()
        assert focals.shape == extra_dims

        # get camera centers and rots
        centers = poses[...,:3,3]
        rots = poses[...,:3,:3]

        # create base rays, [3,res,res]
        image_plane_size = 1/focals
        pixel_size = image_plane_size/resolution
        max_pixel_pos = (image_plane_size-pixel_size)/2
        pixel_positions = torch.linspace(-1,1,resolution).to(focals.device)
        pixel_grid_x, pixel_grid_y = torch.meshgrid(pixel_positions,-pixel_positions,indexing='xy')
        pixel_grid_x = pixel_grid_x[(None,)*n_extra_dims] * max_pixel_pos[...,None,None]
        pixel_grid_y = pixel_grid_y[(None,)*n_extra_dims] * max_pixel_pos[...,None,None]
        base_dirs = torch.stack([pixel_grid_x,pixel_grid_y,-torch.ones_like(pixel_grid_x)],axis=-3)
        base_dirs_flat = base_dirs.view(*extra_dims,3,-1)
        base_dirs_flat /= torch.linalg.norm(base_dirs_flat,2,-2,keepdim=True)

        # transform base dirs to create rays
        dirs = rots @ base_dirs_flat
        dirs = dirs.reshape(*poses.shape[:-2],3,resolution,resolution)

        # add centers to make ray
        rays = torch.cat([dirs,centers[...,:,None,None].tile((resolution,resolution))],axis=-3)

        return rays

    def freq_enc(self,rays):
        """
        Frequency encodes rays.
        :param rays: a [* x 6 x H x W] ray representation.
        :return: a [* x R x H x W] encoded ray representation, R is defined as self.n_out_channels
        """
        extra_dims = rays.shape[:-3]
        n_extra_dims = len(extra_dims)

        # encodeing does not repeat until [-half_period, half_period]
        n_in_channels = rays.shape[-3]
        frequency_exponent = torch.arange(self.n_frequencies)
        frequency_multiplier = (2.0**frequency_exponent)/self.half_period
        frequency_multiplier = frequency_multiplier[:,None].tile(1,n_in_channels).view(-1)[:,None,None][(None,)*n_extra_dims]*torch.pi
        rays_tiled = rays.tile(self.n_frequencies,1,1)*frequency_multiplier.to(rays.device)
        rays_sin = torch.sin(rays_tiled)
        rays_cos = torch.cos(rays_tiled)
        fourier_feats = torch.cat([rays,rays_sin,rays_cos],-3)
        return fourier_feats

    def forward(self,poses,focals):
        """
        :param poses: poses as 4x4 matricies [B, N, 4, 4]
        :param focals: focal lengths [B, N]
        :return: a dict of encoded rays at various resolutions, each resolution contains rays for N views in their own canonical reference frame, and rays of every N view in the reference frame of every other view.
        """
        assert poses.shape[0] == focals.shape[0]
        assert poses.shape[1] == focals.shape[1]
        batch_size = poses.shape[0]
        n_views = poses.shape[1]

        # compute relative poses, [B, N, N, 4, 4]
        tiled_q_poses = poses[:,:,None,:,:].tile((n_views,1,1))
        tiled_k_poses = poses[:,None,:,:,:].tile((n_views,1,1,1))
        rel_poses = torch.linalg.solve(tiled_q_poses,tiled_k_poses)
        q_poses = torch.eye(4,device=rel_poses.device)[None,None,:,:].tile((batch_size,n_views,1,1))

        # convert to rays
        out = {}
        for resolution in self.resolutions:
            q_rays = self.to_rays(q_poses,focals,resolution)
            k_rays = self.to_rays(rel_poses,focals[:,None,:].tile((n_views,1)),resolution)

            # encode
            q_encoded = self.freq_enc(q_rays) # [B x N x R x H x W]
            k_encoded = self.freq_enc(k_rays) # [B x N1 x N2 x R x H x W], N == N1 == N2, each N2 view is relative to each N1 view

            out[resolution] = {'q':q_encoded,'k':k_encoded}

        return out
