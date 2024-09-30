import lightning.pytorch as pl
import torch
from utils import instantiate_from_config
from models.ddpm.DDPM import DDPM

class LDM(DDPM):
    def __init__(self,latent_autoencoder_config,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.latent_autoencoder = instantiate_from_config(latent_autoencoder_config)

    def training_step(self, batch, batch_idx):
        encoded_batch = self.latent_autoencoder.encode_to_prequant(batch['im'])
        encoded_batch = {'im':encoded_batch}
        return super().training_step(encoded_batch, batch_idx)

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        assert return_intermediates == False, 'return_intermediates not implemented'
        n_enc_downsamples = self.latent_autoencoder.encoder.num_resolutions - 1
        down_factor = 2**n_enc_downsamples
        assert shape[2]%down_factor == 0, 'shape not factor of 2'
        assert shape[3]%down_factor == 0, 'shape not factor of 2'
        latent_shape = shape[:2]+[shape[2]//down_factor,shape[3]//down_factor]
        encoded_im = super().p_sample_loop(latent_shape,return_intermediates)
        decoded_im = self.latent_autoencoder.decode_code(encoded_im)
        return decoded_im
