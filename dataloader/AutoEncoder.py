# AutoEncoder.py

import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.flc = 32
        self.zdim = 512
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.flc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.encoder.add_module("final_convs", nn.Sequential(
            nn.Conv2d(self.flc, self.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*4, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.zdim, kernel_size=8, stride=1, padding=0)
        ))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.zdim, self.flc, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.flc*4, self.flc*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc*2, self.flc*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.flc*2, self.flc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.flc, self.flc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(self.flc, self.flc, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(self.flc, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x2 = self.decoder(x1)
        return x2
