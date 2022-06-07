import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

import math

class CCNN(nn.Module):
    """Network for confidence
    -Since it uses features of depth encoder, it should have same architecture with depth encoder Encoder_Q.
    """
    def __init__(self):
        super(CCNN, self).__init__()
        self.num_classes = 19
        self.cost_conv1 = nn.Conv2d(self.num_classes, 64, kernel_size=9, padding=4)
        self.cost_bn1 = nn.BatchNorm2d(64)
        self.cost_conv2 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        self.cost_bn2 = nn.BatchNorm2d(64)
        self.cost_conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.cost_bn3 = nn.BatchNorm2d(64)
        self.cost_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.cost_bn4 = nn.BatchNorm2d(64)
        self.cost_pred = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, disp):
        x = F.relu(self.cost_bn1(self.cost_conv1(disp)))
        x = F.relu(self.cost_bn2(self.cost_conv2(x)))
        x = F.relu(self.cost_bn3(self.cost_conv3(x)))
        x = F.relu(self.cost_bn4(self.cost_conv4(x)))
        out = self.sigmoid(self.cost_pred(x))

        return out


class ThresNet(nn.Module):
    def __init__(self, class_adaptive=False):
        super(ThresNet, self).__init__()
        if not class_adaptive:
            self.input_c = 1
            self.output_c = 1
        else:
            self.input_c = 19   # need to check!
            self.output_c = 19
        self.cost_conv1 = nn.Conv2d(self.input_c, 64, stride=2, kernel_size=9, padding=4)   # can we convolve on B x C feature?
        self.cost_bn1 = nn.BatchNorm2d(64)
        self.cost_conv2 = nn.Conv2d(64, 64, stride=2, kernel_size=7, padding=3)
        self.cost_bn2 = nn.BatchNorm2d(64)
        self.cost_conv3 = nn.Conv2d(64, 64, stride=2, kernel_size=5, padding=2)
        self.cost_bn3 = nn.BatchNorm2d(64)
        self.cost_conv4 = nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1)
        self.cost_bn4 = nn.BatchNorm2d(64)
        #self.cost_pred = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.fc8 = nn.Linear(64, self.output_c)
        self.activation = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, confidence_map):
        x = F.relu(self.cost_bn1(self.cost_conv1(confidence_map)))
        x = F.relu(self.cost_bn2(self.cost_conv2(x)))
        x = F.relu(self.cost_bn3(self.cost_conv3(x)))
        x = F.relu(self.cost_bn4(self.cost_conv4(x)))
        x = F.adaptive_avg_pool2d(x,(1, 1))

        x = x.view(confidence_map.shape[0], 64)
        x = self.fc8(x)

        x = self.activation(x)

        return x


class ThresNet_P(nn.Module):
    def __init__(self, class_adaptive=True):
        super(ThresNet_P, self).__init__()
        self.input_c = 19   # need to check!
        self.output_c = 19

        self.num_classes = 19
        self.cost_conv1 = nn.Conv2d(self.num_classes, 64, kernel_size=9, padding=4)
        self.cost_bn1 = nn.BatchNorm2d(64)
        self.cost_conv2 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        self.cost_bn2 = nn.BatchNorm2d(64)
        self.cost_conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.cost_bn3 = nn.BatchNorm2d(64)
        self.cost_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.cost_bn4 = nn.BatchNorm2d(64)
        self.cost_conv5 = nn.Conv2d(64, self.num_classes, kernel_size=3, padding=1)
        self.cost_bn5 = nn.BatchNorm2d(self.num_classes)
        # self.cost_pred = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.cost_pred = nn.Conv2d(19, 1, kernel_size=1, padding=0)


        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, prob, class_onehot):
        B, C, H, W = prob.shape


        x1 = F.relu(self.cost_bn1(self.cost_conv1(prob)))
        x2 = F.relu(self.cost_bn2(self.cost_conv2(x1)))
        x3 = F.relu(self.cost_bn3(self.cost_conv3(x2)))
        x4 = F.relu(self.cost_bn4(self.cost_conv4(x3)))
        x5 = F.relu(self.cost_bn5(self.cost_conv5(x4))) # 8 19 64 128

        # class adaptive pooling
        x6 = torch.stack([x5]*C, dim=2).view(B, C, C, H, W)
        class_onehot = torch.stack([class_onehot]*C,dim=1).view(B, C, C, H, W)
        x7 = x6 * class_onehot
        x8 = x7.view(B, C * C, -1)
        x9 = x8.view(B, C, C, -1)

        class_onehot = class_onehot.view(B, C * C, -1)
        class_onehot = class_onehot.view(B, C, C, -1)

        x10 = (x9.sum(dim=3) / (class_onehot.sum(dim=3) + 1e-10)).unsqueeze(3)

        out = self.sigmoid(self.cost_pred(x10))
        out = out.squeeze(3).squeeze(1)

        return out


