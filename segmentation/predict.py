class SegNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(SegNet, self).__init__()


        # encoder (downsampling)

        self.enc_conv0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.norm0 = nn.BatchNorm2d(32)
        self.enc_conv0_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)  # 256 -> 128

        self.enc_conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.enc_conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 128 -> 64

        self.enc_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm2d(128)
        self.enc_conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm6 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 64 -> 32

        self.enc_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.norm7 = nn.BatchNorm2d(256)
        self.enc_conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm8 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # bottleneck
        self.bottleneck_conv =  nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.norm9 = nn.BatchNorm2d(256)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear') # 16 -> 32
        self.dec_conv0 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.norm10 = nn.BatchNorm2d(128)
        self.dec_conv0_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.norm11 = nn.BatchNorm2d(128)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear'  )# 32 -> 64
        self.dec_conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.norm12 = nn.BatchNorm2d(64)
        self.dec_conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.norm13 = nn.BatchNorm2d(64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear'  )# 32 -> 64
        self.dec_conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.norm14 = nn.BatchNorm2d(32)
        self.dec_conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.norm15 = nn.BatchNorm2d(32)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear'  )# 32 -> 64
        self.dec_conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
        self.norm16 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.norm0(F.relu(self.enc_conv0(x)))
        x = self.norm1(F.relu(self.enc_conv0_1(x)))
        x = self.pool0(x)

        x = self.norm2(F.relu(self.enc_conv1(x)))
        x = self.norm3(F.relu(self.enc_conv1_1(x)))
        x = self.pool1(x)

        x = self.norm5(F.relu(self.enc_conv2(x)))
        x = self.norm6(F.relu(self.enc_conv2_1(x)))
        x = self.pool2(x)

        x = self.norm7(F.relu(self.enc_conv3(x)))
        x = self.norm8(F.relu(self.enc_conv3_1(x)))
        x = self.pool3(x)

        # bottleneck
        x = self.norm9(self.bottleneck_conv(x))

        # decoder (upsampling)
        x = self.upsample0(x)
        x = self.norm10(F.relu(self.dec_conv0(x)))
        x = self.norm11(F.relu(self.dec_conv0_1(x)))

        x = self.upsample1(x)
        x = self.norm12(F.relu(self.dec_conv1(x)))
        x = self.norm13(F.relu(self.dec_conv1_1(x)))
        x = self.upsample2(x)
        x = self.norm14(F.relu(self.dec_conv2(x)))
        x = self.norm15(F.relu(self.dec_conv2_1(x)))

        x = self.upsample3(x)
        x =self.norm16(self.dec_conv3(x))
        return x

#Ещё одна вариация сегнета с заменой UpSample на Unpool
class SegNet2(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder (downsampling)

        self.x0 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2, dilation=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(32),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(32))
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
        self.y0 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=8, dilation=8),
                                nn.ReLU(),
                                nn.BatchNorm2d(128),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(128))

        self.unpool1 = nn.MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.y1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=4, dilation=4),
                                nn.ReLU(),
                                nn.BatchNorm2d(64),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(64))

        self.unpool2 = nn.MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.y2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=2, dilation=2),
                                nn.ReLU(),
                                nn.BatchNorm2d(32),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.BatchNorm2d(32))

        self.unpool3 = nn.MaxUnpool2d(2, stride=2)  # 32 -> 64
        self.y_final = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2, dilation=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(32),
                                     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(16),
                                     nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1))

    def forward(self, a):
        a = self.x0(a)
        a, indices0 = self.pool0(a)
        a = self.x1(a)
        a, indices1 = self.pool1(a)
        a = self.x2(a)
        a, indices2 = self.pool2(a)
        a = self.x3(a)
        a, indices3 = self.pool3(a)
        a = self.unpool0(a, indices3)
        a = self.y0(a)
        a = self.unpool1(a, indices2)
        a = self.y1(a)
        a = self.unpool2(a, indices1)
        a = self.y2(a)
        a = self.unpool2(a, indices0)
        a = self.y_final(a)
        return a




