import os
import hydra
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from src import model
from src.dataloader import get_mnist_dataloaders

AVAIL_GPUS = min(1, torch.cuda.device_count())


def trainGAN(G, D, dataloader, opt_g, opt_d, logger, cfg):

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
        valid_z = torch.randn(16, cfg.latent_dim)
        z = valid_z.type_as(generator.model[0].weight)

        # log sampled images
        sample_imgs = generator(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        logger.add_image("generated_val_images", grid, current_epoch)

    G.eval()
    D.eval()
    for e in range(cfg.epoches):
        minG_steps = 0
        maxD_steps = 0
        for i, batches in tqdm(enumerate(dataloader)):
            imgs = batches[0]
            # sample noise
            z = torch.randn(imgs.shape[0], cfg.latent_dim)
            z = z.type_as(imgs)

            # max Discriminator 
            if i % cfg.maxD_steps == cfg.maxD_steps-1:
                G.eval()
                D.train()
                # how well can it label as real?
                valid = torch.ones(imgs.size(0), 1)
                valid = valid.type_as(imgs)
                real_loss = adversarial_loss(D(imgs), valid)

                # how well can it label as fake?
                fake = torch.zeros(imgs.size(0), 1)
                fake = fake.type_as(imgs)
                fake_loss = adversarial_loss(D(G(z).detach()), fake)

                # discriminator loss is the average of these
                d_loss = (real_loss + fake_loss) / 2

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                logger.add_scalar('d_loss', d_loss, maxD_steps)
                maxD_steps += 1

            # min Generator
            else:
                G.train()
                D.eval()   
                
                # ground truth result (ie: all fake)
                # put on GPU because we created this tensor inside training_loop
                valid = torch.ones(imgs.size(0), 1)
                valid = valid.type_as(imgs)

                # adversarial loss is binary cross-entropy
                g_loss = adversarial_loss(D(G(z)), valid)

                opt_g.zero_grad()
                g_loss.backward()
                opt_g.step()

                log_generated_images(G, z, logger, minG_steps, num=6)
                logger.add_scalar('g_loss', g_loss, minG_steps)
                minG_steps += 1

        on_epoch_end(G, e, cfg)


@hydra.main(config_path="./cfg", config_name='config')
def main(cfg):
    torch.manual_seed(cfg.seed)
    orig_cwd = hydra.utils.get_original_cwd()
    train_loader, _ = get_mnist_dataloaders(orig_cwd, cfg.img_size, cfg.batch_size)

    logger = SummaryWriter()
    #logger = None

    generator = model.Generator(cfg)
    discriminator = model.Discriminator(cfg)

    if cfg.optimizer == 'adam':
        opt_g = torch.optim.Adam(generator.parameters(),     lr=cfg.lr_g)
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=cfg.lr_d)
    elif cfg.optimizer == 'rmsprop':
        opt_g = torch.optim.RMSprop(generator.parameters(),     lr=cfg.lr_g)
        opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=cfg.lr_d)


    trainGAN(G=generator, D=discriminator, dataloader=train_loader, opt_g=opt_g, opt_d=opt_d, logger=logger, cfg=cfg)


if __name__ == '__main__':
    main()