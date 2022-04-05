import torch
import torch.nn.functional as F
from torch import nn
import hydra
from src.base_block import *

class VAE(nn.Module):
    def __init__(self, cfg):
        super(VAE, self).__init__()

        self.latent_dim = cfg.model.latent_dim
        self.kld_weight = cfg.model.kld_weight
        self.img_size   = cfg.img_size
        self.loss_func  = cfg.model.loss
        self.model_arc  = cfg.model.arch
        if self.model_arc == 'deconv':
            self.encoder = Encoder(cfg)
            self.decoder = Decoder(cfg)
        elif self.model_arc == 'mlp':
            self.encoder = nn.Sequential(
                            nn.Linear(self.img_size ** 2, 8 ** 2),
                            nn.ReLU(),
                            nn.Linear(8 ** 2, 4 ** 2),
                            nn.ReLU(),
                            nn.Linear(4 ** 2,  2 * self.latent_dim)
            )

            self.decoder = nn.Sequential(
                            nn.Linear(self.latent_dim,  4 **2 ),
                            nn.ReLU(),
                            nn.Linear(4 **2, 8 ** 2),
                            nn.ReLU(),
                            nn.Linear(8 ** 2, self.img_size ** 2),
                            nn.Sigmoid(),
            )
        else:
            raise NotImplementedError


    def forward(self, batch):
        if self.model_arc == 'deconv':
            mu, log_var = self.encoder(batch)
            mu, log_var = mu.reshape(-1, self.latent_dim), log_var.reshape(-1, self.latent_dim)
            latent_codes = self.reparameterize(mu, log_var)
            recons = self.decoder(latent_codes)
        elif self.model_arc == 'mlp':
            mu, log_var = torch.chunk(self.encoder(batch.flatten(1)), 2, dim=1)
            latent_codes = self.reparameterize(mu, log_var)
            recons = self.decoder(latent_codes).reshape(-1, 1, self.img_size, self.img_size)

        return mu, log_var, recons

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss(self, input, mu, log_var, recons):
        ''' Reconstruction loss + KL divergence
        Here KL( N(mu, sigma) | N(0,I)) = 0.5 * ( mu^T * mu + Trace(sigma) - d - log(sigma))
        '''
        if self.loss_func == 'bce':
            recons_loss = F.binary_cross_entropy(recons.flatten(1), input.flatten(1))
        elif self.loss_func == 'mse':
            recons_loss =F.mse_loss(recons, input)
        else:
            raise NotImplementedError(self.loss_func + ' is not implemented now')

        kld_loss = 0.5 * ((mu ** 2 + log_var.exp() - log_var).sum(dim=-1) - self.latent_dim).mean(dim=0)
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.clone().detach(), 'KLD':-kld_loss.clone().detach()}

    def generate(self, latents):
        ''' 
        latents: [B * latent_dim]
        '''
        return self.decoder(latents).reshape(-1, 1, self.img_size, self.img_size) 


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.latent_dim = cfg.model.latent_dim
        self.img_size   = cfg.img_size
        self.channel    = cfg.channel


        self.latent_to_features = nn.Sequential(
            nn.Linear(self.latent_dim, 16 * self.channel * 1 * 1),
            nn.BatchNorm1d(16 * self.channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16 * self.channel, 64 * self.channel * 1 * 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(64 * self.channel),
        )


        self.hidden_dims =  self.channel * [64, 128, 64, 32, 8, 1]
        features_to_image = []
        for in_dim, out_dim in zip(self.hidden_dims[:-2], self.hidden_dims[1:-1]):
            features_to_image += generator_block(in_dim, out_dim, bn=True)
        features_to_image += [ nn.ConvTranspose2d(self.hidden_dims[-2], self.hidden_dims[-2], 4, 2, 1),
                                nn.BatchNorm2d( self.hidden_dims[-2]),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(self.hidden_dims[-2], self.hidden_dims[-1], 3, 1, 1),
                               #nn.Sigmoid() 
                               nn.Tanh() 
                               ]
        
        self.features_to_image = nn.Sequential(*features_to_image)

    def forward(self, batch):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(batch)
        # Reshape
        x = x.view(-1, 64 * self.channel, 1, 1) 
        # Return generated image
        return self.features_to_image(x)
    
 
class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.latent_dim = cfg.model.latent_dim
        self.channel = cfg.channel

        self.hidden_dims =  self.channel * [1, 16, 32, 64, 128] +  [self.latent_dim * 2] 
        image_to_feature = []
        for in_dim, out_dim in zip(self.hidden_dims[:-2], self.hidden_dims[1:-1]):
            image_to_feature += discriminator_block(in_dim, out_dim, bn=True)
        image_to_feature += [ nn.Conv2d(self.hidden_dims[-2], self.hidden_dims[-1], 3, 2, 1),
                              ]
        
        self.image_to_latent = nn.Sequential(*image_to_feature) 

    def forward(self, batch):
        # return std and logvar respectively
        return torch.chunk(self.image_to_latent(batch), 2, dim=1)


@hydra.main(config_path="../cfg", config_name='config')
def test(cfg):
    model = VAE(cfg)

    img = torch.randn((2,1,32,32))
    mu, logvar, out = model(img)
    print(out.shape)
    print("mu = {}, log_var = {}".format(mu.shape, logvar.shape))

if __name__ == '__main__':
    test()
