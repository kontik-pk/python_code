#Unet with dilated convolutions

class DilatedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder (downsampling)

        self.x0 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2, dilation=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(32),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(32), )
        self.pool0 = nn.MaxPool2d(2, stride=2, return_indices=True)  # 256 -> 128

        self.x1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=4, dilation=4),
                                nn.ReLU(),
                                nn.BatchNorm2d(64),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(64))
        self.pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)  # 128 -> 64

        self.x2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=8, dilation=8),
                                nn.ReLU(),
                                nn.BatchNorm2d(128),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(128))
        self.pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)  # 64 -> 32
        self.x3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=16, dilation=16),
                                nn.ReLU(),
                                nn.BatchNorm2d(256),
                                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(256))
        self.pool3 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # decoder (upsampling)
        self.unpool0 = nn.MaxUnpool2d(2, stride=2)  # 16 -> 32
        self.y0 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=8, dilation=8),
                                nn.ReLU(),
                                nn.BatchNorm2d(128),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(128))

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.y1 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=4, dilation=4),
                                nn.ReLU(),
                                nn.BatchNorm2d(128),
                                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(64))

        self.unpool2 = nn.MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.y2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=2, dilation=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(64),
                                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(32))

        self.unpool3 = nn.MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.y_final = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=2, dilation=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1))

    def forward(self, a):
        a = self.x0(a)
        m = a.clone()
        a, indices0 = self.pool0(a)
        a = self.x1(a)
        l = a.clone()
        a, indices1 = self.pool1(a)
        a = self.x2(a)
        k = a.clone()
        a, indices2 = self.pool2(a)
        a = self.x3(a)
        a_c = a.clone()
        a, indices3 = self.pool3(a)
        a = self.unpool0(a, indices3)
        a = torch.cat((a, a_c), 1)
        a = self.y0(a)
        a = self.unpool1(a, indices2)
        a = torch.cat((a, k), 1)
        a = self.y1(a)
        a = self.unpool2(a, indices1)
        a = torch.cat((a, l), 1)
        a = self.y2(a)
        a = self.unpool2(a, indices0)
        a = torch.cat((a, m), 1)
        a = self.y_final(a)
        return a
