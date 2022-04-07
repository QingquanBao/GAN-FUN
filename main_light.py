import os
import hydra
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from src import vanilla_gan
from src.dataloader import get_mnist_dataloaders

AVAIL_GPUS = min(1, torch.cuda.device_count())

class GAN(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.generator = vanilla_gan.Generator(cfg)
        self.discriminator = vanilla_gan.Discriminator(cfg)

    def forward(self, z):   
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.cfg.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.cfg.lr_g
        #b1 = self.cfg.b1
        #b2 = self.cfg.b2

        #opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        #opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        valid_z = torch.randn(16, self.hparams.latent_dim)
        z = valid_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

    def train_dataloader(self):
        train_loader, _ = get_mnist_dataloaders(self.cfg.img_size, self.cfg.batch_size)
        return train_loader

@hydra.main(config_path="./cfg", config_name='config')
def main(cfg):
    #torch.manual_seed(cfg.seed)
    pl.seed_everything(cfg.seed)

    model = GAN(cfg)
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=5,
        progress_bar_refresh_rate=10,
        precision=32,
        log_every_n_steps=10
    )

    trainer.fit(model)
    #trainer.test(model)

if __name__ == '__main__':
    main()