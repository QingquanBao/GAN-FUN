import torch
import torch.nn.functional as F
from torch import nn
import hydra
from src.base_block import *

class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg
        self.latent_dim = self.cfg.model.latent_dim
        self.img_size   = self.cfg.img_size
        self.channel    = self.cfg.channel

        self.feature_sizes = (int(self.img_size / 16), int(self.img_size / 16))

        #self.latent_to_features = nn.Sequential(
        #    nn.Linear(self.latent_dim, 128 * self.channel * self.feature_sizes[0] * self.feature_sizes[1]),
        #    nn.LeakyReLU(0.2, inplace=True)
        #)


        self.hidden_dims = [self.latent_dim] + self.channel * [128, 64, 32, 16, 1]
        features_to_image = []
        for in_dim, out_dim in zip(self.hidden_dims[:-2], self.hidden_dims[1:-1]):
            features_to_image += generator_block(in_dim, out_dim, bn=True)
        features_to_image += [ nn.ConvTranspose2d(self.hidden_dims[-2], self.hidden_dims[-1], 4, 2, 1),
                               nn.Tanh() ]
        #features_to_image += [ nn.ConvTranspose2d(self.hidden_dims[-2], 8, 4, 2, 1),
        #                    nn.BatchNorm2d(8, 0.8),
        #                    nn.LeakyReLU(0.2, inplace=True),
        #                    nn.Conv2d(8 * self.channel, self.cfg.channel, 3, stride=1, padding=1),
        #                       nn.Tanh() ]
        
        self.features_to_image = nn.Sequential(*features_to_image)
                                 

        '''
        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(self.cfg.latent_dim, 128 * self.cfg.channel, 4, 2, 1),
            nn.BatchNorm2d(128 * self.cfg.channel),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.ConvTranspose2d(128 * self.cfg.channel, 64 * self.cfg.channel, 4, 2, 1),
            nn.BatchNorm2d(64 * self.cfg.channel),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU()
            nn.ConvTranspose2d(64 * self.cfg.channel, 32 * self.cfg.channel, 4, 2, 1),
            nn.BatchNorm2d(32 * self.cfg.channel),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(),
            nn.ConvTranspose2d(32 * self.cfg.channel, 16 * self.cfg.channel, 4, 2, 1),
            nn.BatchNorm2d(16 * self.cfg.channel),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.ReLU(),
            nn.ConvTranspose2d(16 * self.cfg.channel, 1 * self.cfg.channel , 4, 2, 1),
            #nn.BatchNorm2d(8 * self.cfg.channel),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(8 * self.cfg.channel, self.cfg.channel, 3, stride=1, padding=1),
            nn.Tanh()
        )
        '''

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        #x = self.latent_to_features(input_data)
        # Reshape
        #x = x.view(-1, 128 * self.cfg.channel, self.feature_sizes[0], self.feature_sizes[1])
        x = input_data.view(-1, input_data.shape[1], 1, 1)
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
    '''

class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.init_size = cfg.img_size // 4
        self.latent_dim = cfg.latent_dim
        self.l1 = nn.Sequential(nn.Linear(cfg.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, cfg.channel, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))
'''

class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg
        self.model = LeNet()

    def forward(self, batch):
        return self.model(batch)
        #return torch.sigmoid(self.model(batch))

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class LeNet(nn.Module):
    # copy from PyTorch Tutorial
    def __init__(self):
        super(LeNet, self).__init__()
        '''
        # 1 input image channel (black & white), 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 7)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 1)
        '''

        self.conv = nn.Sequential(
        nn.Conv2d(1, 6, 7),
        nn.LeakyReLU(0.2, inplace=True),
        #nn.MaxPool2d((2,2)),
        nn.AvgPool2d((2,2)),
        nn.Conv2d(6, 16, 3),
        nn.LeakyReLU(0.2, inplace=True),
        nn.AvgPool2d(2),
        #nn.MaxPool2d(2),
        )

        self.linear = nn.Sequential(
        nn.Linear(16 * 5 * 5, 120),  # 6*6 from image dimension
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(120, 84),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(84, 20),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(20, 1),
        nn.Sigmoid()
        )
    def forward(self, x):
        '''
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        '''
        x = self.conv(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.linear(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


@hydra.main(config_path="../cfg", config_name='config')
def test(cfg):
    G = Generator(cfg)
    D = Discriminator(cfg)

    img = torch.randn((1,1,32,32))
    out = D(img)
    print(out.shape)

    hidden = torch.randn((1, 16))
    out_img = G(hidden)

    print(out_img.shape)


if __name__ == '__main__':
    test()


