import lightning.pytorch as pl
import torch
from utils import instantiate_from_config
from models.ldm import LDM
from models.multiview_unet.ray_encoder import Ray_encoder
from torch.nn import functional as F
from tqdm import tqdm

def extract_into_tensor(a, t, x_shape):
    b, n, *_ = t.shape
    out = a.gather(-1, t.view(-1))
    return out.reshape(b, n, *((1,) * (len(x_shape) - 2)))

class Multiview_LDM(LDM):
    def __init__(self,ray_resolutions,ray_frequencies,ray_half_period,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.ray_encoder = Ray_encoder(ray_resolutions,n_frequencies=ray_frequencies,half_period=ray_half_period)

    def training_step(self, batch, batch_idx):
        # prep inputs
        batch_size, n_views, _, *spatial = batch['ims'].shape
        stacked_ims = batch['ims'].view(-1,3,*spatial)
        encoded_im = self.latent_autoencoder.encode_to_prequant(stacked_ims)
        _,_,*encoded_spatial = encoded_im.shape
        encoded_im = encoded_im.view(batch_size, n_views, 3, *encoded_spatial)
        x_start = encoded_im
        context = self.ray_encoder(batch['poses'],batch['focals'])

        # add valid views  to context
        view_mask = batch['view_mask']
        context['view_mask'] = view_mask

        # choose some views as conditioning
        n_valid_views = view_mask.sum(1)
        n_cond_views = torch.stack([torch.randint(0,x,()) for x in n_valid_views])
        unshuffled_cond_mask = [torch.tensor([True]*(y)+[False]*(x-y)) for x,y in zip(n_valid_views,n_cond_views)]
        shuffled_cond_mask = [x[torch.randperm(x.shape[0])] for x in unshuffled_cond_mask]
        cond_mask_full = torch.stack([F.pad(x,(0,n_views-y)) for x,y in zip(shuffled_cond_mask,n_valid_views)])
        cond_mask_full = cond_mask_full.to(self.device)
        t = torch.randint(0, self.num_timesteps, (batch_size, 1), device=self.device).long()
        t = t*(~cond_mask_full)

        # forward
        noise = torch.randn_like(x_start)*(~cond_mask_full[:,:,None,None,None])
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.denoiser(x_noisy, t, context=context)

        target = noise

        # compute loss
        gen_view_mask = view_mask*(~cond_mask_full)
        loss = torch.nn.functional.mse_loss(target, model_out, reduction='none')
        loss = loss.mean(dim=[2,3,4])
        loss = (loss*gen_view_mask).sum()/gen_view_mask.sum()
        self.log("train_loss", loss)
        return loss

    def p_mean_variance(self, x, t, context, clip_denoised: bool):
        model_out = self.denoiser(x, t, context=context)
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        if clip_denoised: x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, context, clip_denoised=True):
        b, n, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, context=context, clip_denoised=clip_denoised)
        noise = torch.randn(x.shape, device=device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, n, *((1,) * (len(x.shape) - 2)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, poses, focals, return_intermediates=False, print_progress=False):
        assert return_intermediates == False, 'return_intermediates not implemented'
        n_enc_downsamples = self.latent_autoencoder.encoder.num_resolutions - 1
        down_factor = 2**n_enc_downsamples
        assert shape[3]%down_factor == 0, 'shape not factor of 2'
        assert shape[4]%down_factor == 0, 'shape not factor of 2'
        latent_shape = shape[:3]+[shape[3]//down_factor,shape[4]//down_factor]

        # encode rays
        context = self.ray_encoder(poses,focals)

        # sample loop
        device = self.betas.device
        b,n = shape[0],shape[1]
        encoded_im = torch.randn(latent_shape, device=device)
        step_iter = tqdm(list(reversed(range(0, self.num_timesteps)))) if print_progress else reversed(range(0, self.num_timesteps))
        for i in step_iter:
            t = torch.full((b,n), i, device=device, dtype=torch.long)
            encoded_im = self.p_sample(encoded_im, t, context=context, clip_denoised=True)

        # decode image
        batch_size, n_views, _, *encoded_spatial = encoded_im.shape
        stacked_encoded = encoded_im.view(-1,3,*encoded_spatial)
        decoded_im = self.latent_autoencoder.decode_code(stacked_encoded)
        decoded_im = decoded_im.view(batch_size,n_views,3,shape[3],shape[4])
        return decoded_im

    def p_sample_loop_cond(self, cond, shape, poses, focals, return_latents=False, print_progress=False):
        n_enc_downsamples = self.latent_autoencoder.encoder.num_resolutions - 1
        down_factor = 2**n_enc_downsamples
        assert shape[3]%down_factor == 0, 'shape not factor of 2'
        assert shape[4]%down_factor == 0, 'shape not factor of 2'
        latent_shape = shape[:3]+[shape[3]//down_factor,shape[4]//down_factor]

        # encode rays
        context = self.ray_encoder(poses,focals)

        # encoded cond:
        n_cond = cond.shape[1]
        n_generating = shape[1] - n_cond

        # sample loop
        device = self.betas.device
        b,n = shape[0],shape[1]
        encoded_im = torch.randn(latent_shape, device=device)
        step_iter = tqdm(list(reversed(range(0, self.num_timesteps)))) if print_progress else reversed(range(0, self.num_timesteps))
        for i in step_iter:
            t = torch.full((b,n), i, device=device, dtype=torch.long)
            t[:,:n_cond] = 0 # assume cond images are placed first
            encoded_im[:,:n_cond,...] = cond # substitute conditioning image
            encoded_im = self.p_sample(encoded_im, t, context=context, clip_denoised=True)

        # decode image
        batch_size, n_views, _, *encoded_spatial = encoded_im.shape
        stacked_encoded = encoded_im.view(-1,3,*encoded_spatial)
        decoded_im = self.latent_autoencoder.decode_code(stacked_encoded)
        decoded_im = decoded_im.view(batch_size,n_views,3,shape[3],shape[4])
        if return_latents:
            return decoded_im, encoded_im
        else:
            return decoded_im

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        '''
        Samples from the forward noising process
        '''
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
