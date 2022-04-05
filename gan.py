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

from src import vanilla_gan
from src.dataloader import get_mnist_dataloaders

AVAIL_GPUS = min(1, torch.cuda.device_count())


def trainGAN(G, D, dataloader, opt_g, opt_d, logger, cfg, device):

    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def log_generated_images(G, z, logger, step, num=6):
        # generate images
        generated_imgs = G(z)

        # log sampled images
        sample_imgs = generated_imgs[:num]
        grid = torchvision.utils.make_grid(sample_imgs)
        logger.add_image("generated_images", grid, step)

    def on_epoch_end(generator, current_epoch, cfg):
        valid_z = torch.randn(16, cfg.model.latent_dim).to(device)
        #z = valid_z.type_as(generator.model[0].weight)

        # log sampled images
        sample_imgs = generator(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        logger.add_image("generated_val_images", grid, current_epoch)

    #G.eval()
    #D.eval()
    minG_steps = 0
    maxD_steps = 0
    for e in range(cfg.epoches):
        for i, batches in enumerate(tqdm(dataloader)):
            imgs = batches[0].to(device)
            noise1 = 0.03 * torch.rand(imgs.shape).to(imgs.device)
            noise2 = 0.03 * torch.rand(imgs.shape).to(imgs.device)
            imgs += noise1
            # sample noise
            z = torch.randn(imgs.shape[0], cfg.model.latent_dim).to(device)
            z = z.type_as(imgs)

            # max Discriminator 
            #if i % cfg.maxD_steps != cfg.maxD_steps-1:
                #G.eval()
                #D.train()
            opt_d.zero_grad()
            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)
            real_loss = adversarial_loss(D(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)
            fake_loss = adversarial_loss(D(G(z).detach()+ noise2), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            opt_d.step()

            logger.add_scalar('d_loss', d_loss, maxD_steps)
            maxD_steps += 1

            # min Generator
            #else:
                #G.train()
                #D.eval()   
            opt_g.zero_grad()
            
            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = adversarial_loss(D(G(z)), valid)

            g_loss.backward()
            opt_g.step()

            log_generated_images(G, z, logger, minG_steps, num=6)
            logger.add_scalar('g_loss', g_loss, minG_steps)
            minG_steps += 1

        on_epoch_end(G, e, cfg)


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

    generator = vanilla_gan.Generator(cfg).to(device)
    discriminator = vanilla_gan.Discriminator(cfg).to(device)

    if cfg.model.optimizer == 'adam':
        opt_g = torch.optim.Adam(generator.parameters(),     lr=cfg.model.lr_g, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.model.lr_d, betas=(0.5, 0.999))
    elif cfg.model.optimizer == 'rmsprop':
        opt_g = torch.optim.RMSprop(generator.parameters(),     lr=cfg.model.lr_g)
        opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=cfg.model.lr_d)


    trainGAN(G=generator, D=discriminator, dataloader=train_loader, opt_g=opt_g, opt_d=opt_d, logger=logger, cfg=cfg, device=device)


if __name__ == '__main__':
    main()
