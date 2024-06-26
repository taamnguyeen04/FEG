import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, image_channels, c_dim):
        super(Generator, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(image_channels + c_dim, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            # nn.ReLU(),
            # nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(256),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(256) for i in range(6)]
        )

        self.upsampling = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.InstanceNorm2d(128),
            # nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, image_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, x, target_domain):
        target_domain_expanded = target_domain.unsqueeze(2).unsqueeze(3).expand(-1, -1, 224, 224)
        x = torch.cat((x, target_domain_expanded), dim=1)
        x = self.downsampling(x)
        x = self.bottleneck(x)
        x = self.upsampling(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size, image_channels, c_dim):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU())
        curr_dim = 64

        for i in range(1, 4):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(curr_dim * 2))
            layers.append(nn.LeakyReLU())
            curr_dim *= 2

        kernel_size = int(image_size / 2 ** 4)
        self.model = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.model(x)
        out_src = torch.sigmoid(self.conv1(h))  # Apply sigmoid for binary classification
        out_cls = self.conv2(h)
        out_cls = self.relu(out_cls)  # Apply ReLU activation
        out_cls = torch.softmax(out_cls.view(out_cls.size(0), -1),
                                dim=1)  # Apply softmax for multi-class classification
        return out_src, out_cls