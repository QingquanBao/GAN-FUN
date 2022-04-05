import os
import hydra
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
import random

from src import vanilla_vae
from src.dataloader import get_mnist_dataloaders

AVAIL_GPUS = min(1, torch.cuda.device_count())


def train(model, dataloader, opt, logger, cfg, device):
    def cal_loss(input, mu, log_var, recons):
        ''' Reconstruction loss + KL divergence
        Here KL( N(mu, sigma) | N(0,I)) = 0.5 * ( mu^T * mu + Trace(sigma) - d - log(sigma))
        '''
        recons_loss =F.mse_loss(recons, input)
        #kld_loss = 0.5 * ((mu ** 2 + log_var.exp() - log_var).sum(dim=-1) - self.latent_dim).mean(dim=0)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss +  kld_loss
        return loss

    def log_generated_images(logger, imgs, generated_imgs, steps, num=8):
        # log sampled images
        sample_imgs = generated_imgs[:num]
        grid = torchvision.utils.make_grid(torch.cat([imgs[:num], sample_imgs], dim=0))
        logger.add_image("generated_images", grid, steps)

    def on_epoch_end(model, current_epoch, cfg):
        model.eval()

        num = 20
        if cfg.model.latent_dim == 1:
            latents = (torch.arange(num ** 2) - num**2 / 2).reshape(-1, 1) / 40.
        elif cfg.model.latent_dim == 2:
            indexes = (torch.arange(num) - num / 2 ) / 5.
            x, y = torch.meshgrid(indexes, indexes, indexing='ij')
            latents = torch.stack([x.flatten(), y.flatten()]).T
        else:
            latents = torch.randn(100, cfg.model.latent_dim)
            #raise NotImplementedError
        latents = latents.to(device)

        # log sampled images
        sample_imgs = model.generate(latents)
        grid = torchvision.utils.make_grid(sample_imgs, nrow=num)
        logger.add_image("generated_val_images", grid, current_epoch)

    steps = 0
    for e in range(cfg.epoches):
        for i, batches in enumerate(tqdm(dataloader)):
            model.train()
            opt.zero_grad()

            imgs = batches[0].to(device)
            #noise = 0.03 * torch.rand(imgs.shape).to(imgs.device)
            #imgs += noise

            mu, log_var, out = model(imgs)

            loss_dict = model.loss(imgs, mu, log_var, out)
            loss_dict['loss'].backward()
            #loss = cal_loss(imgs, mu, log_var, out)
            #loss.backward()
            opt.step()

            #logger.add_scalar('loss', loss, steps)
            logger.add_scalar('loss', loss_dict['loss'], steps)
            logger.add_scalar('reconstucion_loss', loss_dict['Reconstruction_Loss'], steps)
            logger.add_scalar('KLD', loss_dict['KLD'], steps)
            log_generated_images(logger, imgs, out, steps, num=8)
            steps += 1

        on_epoch_end(model, e, cfg)


@hydra.main(config_path="./cfg", config_name='config')
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)

    orig_cwd = hydra.utils.get_original_cwd()
    train_loader, _ = get_mnist_dataloaders(orig_cwd, cfg.img_size, cfg.batch_size)

    logger = SummaryWriter()
    #logger = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = vanilla_vae.VAE(cfg).to(device)

    if cfg.model.optimizer == 'adam':
        opt = torch.optim.Adam(model.parameters(),     lr=cfg.model.lr, )
    elif cfg.model.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(model.parameters(),     lr=cfg.model.lr)

    train(model=model, dataloader=train_loader, opt=opt, logger=logger, cfg=cfg, device=device)


if __name__ == '__main__':
    main()