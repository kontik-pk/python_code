
#Создаем дискриминатор
#Эта модель будет принимать на вход снимок и карту, соединять их вместе и пропускать через слои
#На выходе получаем вероятность того, что карта настоящая, а не сгенерированная

class Discriminator(nn.Module):
    def __init__(self, d = 64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(6,d,4,2,1),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(d,d*2, 4,2,1),
                                   nn.BatchNorm2d(d*2),
                                   nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(d*2,d*4, 4,2,1),
                                   nn.BatchNorm2d(d*4),
                                   nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(d*4,d*8, 4,1,1),
                                   nn.BatchNorm2d(d*8),
                                   nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(d*8, 1,4,1,1),
                                   nn.Sigmoid())

    def forward(self, input, label):
      x=torch.cat([input,label],1)
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.conv4(x)
      x = self.conv5(x)
      return x

#Создаем генератор
#Генератор принимает на вход снимок, а возвращает карту
class Generator(nn.Module):
    def __init__(self, d=64):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, d, 4, 2, 1),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(d, d * 2, 4, 2, 1),
                                   nn.BatchNorm2d(d * 2),
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(d * 2, d * 4, 4, 2, 1),
                                   nn.BatchNorm2d(d * 4),
                                   nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(d * 4, d * 8, 4, 2, 1),
                                   nn.BatchNorm2d(d * 8),
                                   nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(d * 8, d * 8, 4, 2, 1),
                                   nn.BatchNorm2d(d * 8),
                                   nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(d * 8, d * 8, 4, 2, 1),
                                   nn.BatchNorm2d(d * 8),
                                   nn.LeakyReLU(0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(d * 8, d * 8, 4, 2, 1),
                                   nn.BatchNorm2d(d * 8),
                                   nn.LeakyReLU(0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(d * 8, d * 8, 4, 2, 1),
                                   nn.ReLU())

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1),
                                     nn.BatchNorm2d(d * 8),
                                     nn.ReLU(),
                                     nn.Dropout(0.5))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1),
                                     nn.BatchNorm2d(d * 8),
                                     nn.ReLU(),
                                     nn.Dropout(0.5))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU(),
                                     nn.Dropout(0.5))
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1),
                                     nn.BatchNorm2d(d * 8),
                                     nn.ReLU())
        self.deconv5 = nn.Sequential(nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1),
                                     nn.BatchNorm2d(d * 4),
                                     nn.ReLU())
        self.deconv6 = nn.Sequential(nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1),
                                     nn.BatchNorm2d(d * 2),
                                     nn.ReLU())
        self.deconv7 = nn.Sequential(nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1),
                                     nn.BatchNorm2d(d),
                                     nn.ReLU())
        self.deconv8 = nn.Sequential(nn.ConvTranspose2d(d * 2, 3, 4, 2, 1),
                                     nn.Tanh())

    def forward(self, x):
        e1 = self.conv1(x)  # 32
        e2 = self.conv2(e1)  # 64
        e3 = self.conv3(e2)  # 128
        e4 = self.conv4(e3)  # 256
        e5 = self.conv5(e4)  # 512
        e6 = self.conv6(e5)  # 512
        e7 = self.conv7(e6)  # 512
        e8 = self.conv8(e7)  # 512

        d1 = self.deconv1(e8)  # 512
        d1 = torch.cat((d1, e7), 1)  # 1024
        d2 = self.deconv2(d1)  # 512
        d2 = torch.cat((d2, e6), 1)  # 1024
        d3 = self.deconv3(d2)  # 512
        d3 = torch.cat((d3, e5), 1)  # 1024
        d4 = self.deconv4(d3)  # 256
        d4 = torch.cat((d4, e4), 1)  # 512
        d5 = self.deconv5(d4)  # 128
        d5 = torch.cat((d5, e3), 1)  # 256
        d6 = self.deconv6(d5)  # 64
        d6 = torch.cat((d6, e2), 1)  # 128
        d7 = self.deconv7(d6)  # 32
        d7 = torch.cat((d7, e1), 1)  # 64
        d8 = self.deconv8(d7)  # 1
        return d8
