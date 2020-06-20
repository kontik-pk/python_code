import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from time import time
from matplotlib import rcParams

#Метрика
#Основана отношении количества пикселей, найденных как в предсказанной маске, так и в исходной маске к количеству пикселей,
# найденных либо в предсказанной маске, либо в исходной
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10

    return thresholded

#Бинарная кросс-энтропия
def bce_loss(y_pred, y_real):
    y_pred = torch.sigmoid(y_pred) #Переведем предсказания в вероятности
    return (y_pred - y_real*y_pred + torch.log(1 + torch.exp

#Расстояние между предсказанной и реальной маской можно измерить с помощью метрики DiceLoss

def dice_loss(y_pred, y_real, smooth=1e-08):
    y_pred = torch.sigmoid(y_pred)
    num = (2 * (y_real * y_pred)).sum()
    den = (y_real + y_pred).sum()
    res = 1 - (num / (den+smooth)) * (1 / (256 * 256))
    return res

#FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, y_pred, y_real):
        y_pred = torch.sigmoid(y_pred) #Переведем предсказания в вероятности
        BCE = y_pred - y_real*y_pred + torch.log(1 + torch.exp(-y_pred))
        pt = torch.exp(-BCE)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE

        return torch.mean(F_loss)

#Structural similarity index
def ssim(x, y, sigma=1.5, L = 255, K1=0.01, K2=0.03):
    c1 = (K1*L)**2
    c2 = (K2*L)**2
    x = torch.sigmoid(x)
    cov = torch.mean((x-torch.mean(x))*(y - torch.mean(y)))
    ssim = (torch.std(x)**2+torch.std(y)**2-2*cov)/(torch.std(x)**2+torch.std(y)**2+c2)
    return ssim






