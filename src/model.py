import torch
import torch.nn.functional as F
from torch import nn
import hydra

class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        self.cfg = cfg

        self.feature_sizes = (int(self.cfg.img_size / 16), int(self.cfg.img_size / 16))

        self.latent_to_features = nn.Sequential(
            nn.Linear(self.cfg.latent_dim, 8 * self.cfg.channel * self.feature_sizes[0] * self.feature_sizes[1]),
            nn.ReLU()
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * self.cfg.channel, 4 * self.cfg.channel, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * self.cfg.channel),
            nn.ConvTranspose2d(4 * self.cfg.channel, 2 * self.cfg.channel, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * self.cfg.channel),
            nn.ConvTranspose2d(2 * self.cfg.channel, self.cfg.channel, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(self.cfg.channel),
            nn.ConvTranspose2d(self.cfg.channel,self.cfg.channel , 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        # Map latent into appropriate size for transposed convolutions
        x = self.latent_to_features(input_data)
        # Reshape
        x = x.view(-1, 8 * self.cfg.channel, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.features_to_image(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.cfg.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()
        self.cfg = cfg
        self.model = LeNet()

    def forward(self, batch):
        return self.model(batch)


class LeNet(nn.Module):
    # copy from PyTorch Tutorial
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 7)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
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


