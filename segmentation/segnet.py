class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder (downsampling)

        self.layer0 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32))
        self.pool0 = nn.MaxPool2d(2, stride=2)  # 256 -> 128

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64))
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 128 -> 64

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128), )
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 64 -> 32

        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256))
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(256))

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2)  # 16 -> 32
        self.dec_conv0 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(256))

        self.upsample1 = nn.Upsample(scale_factor=2)  # 32 -> 64
        self.dec_conv1 = nn.Sequential(nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(128),
                                       nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(128),
                                       nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(128))

        self.upsample2 = nn.Upsample(scale_factor=2)  # 32 -> 64
        self.dec_conv2 = nn.Sequential(nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(64),
                                       nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(64))

        self.upsample3 = nn.Upsample(scale_factor=2)  # 32 -> 64
        self.dec_conv3 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(64),
                                       nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                       nn.ReLU(),
                                       nn.BatchNorm2d(64),
                                       nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.layer0(x)
        conv1 = torch.clone(x)
        x = self.pool0(x)

        x = self.layer1(x)
        conv2 = torch.clone(x)
        x = self.pool1(x)

        x = self.layer2(x)
        conv3 = torch.clone(x)
        x = self.pool2(x)

        x = self.layer3(x)
        conv4 = torch.clone(x)
        x = self.pool3(x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder (upsampling)
        x = torch.cat((self.upsample0(x), conv4), 1)
        x = self.dec_conv0(x)

        x = torch.cat((self.upsample1(x), conv3), 1)
        x = self.dec_conv1(x)

        x = torch.cat((self.upsample2(x), conv2), 1)
        x = self.dec_conv2(x)

        x = torch.cat((self.upsample3(x), conv1), 1)
        x = self.dec_conv3(x)
        return x