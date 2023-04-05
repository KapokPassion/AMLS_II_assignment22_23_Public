import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, scale, inchannels=1):
        super(FSRCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=56, kernel_size=5, stride=1, padding=2, padding_mode='replicate'),
            nn.PReLU()
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(in_channels=56,out_channels=12,kernel_size=1, stride=1, padding=0, padding_mode='replicate'),
            nn.PReLU()
        )

        self.mapping = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride =1, padding=1, padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride =1, padding=1, padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride =1, padding=1, padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride =1, padding=1, padding_mode='replicate'),
            nn.PReLU()
        )

        self.expanding = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=56, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

        # (in_size - 1) * stride - 2 * padding + kernel_size + output_padding = out_size
        if scale == 2:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=56, out_channels=inchannels, kernel_size=9, stride=2, padding=4, output_padding=1, padding_mode='zeros')
            )
        elif scale == 3:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=56, out_channels=inchannels, kernel_size=9, stride=3, padding=3, padding_mode='zeros')
            )
        elif scale == 4:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=56, out_channels=inchannels, kernel_size=9, stride=4, padding=4, output_padding=3, padding_mode='zeros')
            )

    def forward(self, x):
        x = self.features(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)

        return x


class OFSRCNN(nn.Module):
    def __init__(self, scale, inchannels=1):
        super(OFSRCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=56, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(in_channels=56, out_channels=56, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            nn.PReLU()
        )

        self.shrinking = nn.Sequential(
            nn.Conv2d(in_channels=56,out_channels=24, kernel_size=1, stride=1, padding=0, padding_mode='replicate'),
            nn.PReLU()
        )

        self.mapping = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, groups=2, padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, groups=2, padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, groups=2, padding_mode='replicate'),
            nn.PReLU(),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, groups=2, padding_mode='replicate'),
            nn.PReLU()
        )

        self.expanding = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=56, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )
        
        if scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels=56, out_channels=4, kernel_size=3, stride =1, padding=1, padding_mode='replicate'),
                nn.PixelShuffle(2),
                nn.PReLU()
            )
        elif scale == 3:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels=56, out_channels=9, kernel_size=3, stride =1, padding=1, padding_mode='replicate'),
                nn.PixelShuffle(3),
                nn.PReLU()
            )
        elif scale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels=56, out_channels=4, kernel_size=3, stride =1, padding=1, padding_mode='replicate'),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride =1, padding=1, padding_mode='replicate'),
                nn.PixelShuffle(2),
                nn.PReLU()
            )

    def forward(self, x):
        x = self.features(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.upsample(x)

        return x