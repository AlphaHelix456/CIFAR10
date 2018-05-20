import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                              padding=1, bias=False)

        self.drop_rate=drop_rate

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_channels = out_channels * 4
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                              padding=0, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, nblocks, in_channels, growth_rate, block, drop_rate=0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_channels, growth_rate, nblocks, drop_rate)
        
    def _make_layer(self, block, in_channels, growth_rate, nblocks, drop_rate):
        layers = []
        for i in range(nblocks):
            layers.append(block(in_channels+i*growth_rate, growth_rate, drop_rate))
        return nn.Sequential(*layers)
                                
    def forward(self, x):
        return self.layer(x)
                                

class DenseNet(nn.Module):
    def __init__(self, nblocks, growth_rate=12,
                 reduction=0.5, num_classes=10, bottleneck=True, drop_rate=0.0):
        super(DenseNet, self).__init__()
        in_channels = 2 * growth_rate
        if bottleneck:
            block = BottleneckBlock
        else:
            block = BasicBlock
        
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.dense1 = DenseBlock(nblocks[0], in_channels, growth_rate, block, drop_rate)
        in_channels += nblocks[0]*growth_rate
        out_channels = int(math.floor(in_channels*reduction))
        self.trans1 = TransitionBlock(in_channels, out_channels, drop_rate=drop_rate)
        in_channels = out_channels
        # 2nd block
        self.dense2 = DenseBlock(nblocks[1], in_channels, growth_rate, block, drop_rate)
        in_channels += nblocks[1]*growth_rate
        out_chanels = int(math.floor(in_channels*reduction))
        self.trans2 = TransitionBlock(in_channels, out_channels, drop_rate=drop_rate)
        in_channels = out_channels
        # 3rd block
        self.dense3 = DenseBlock(nblocks[2], in_channels, growth_rate, block, drop_rate)
        in_channels += nblocks[2]*growth_rate
        out_channels = int(math.floor(in_channels*reduction))
        self.trans3 = TransitionBlock(in_channels, out_channels, drop_rate=drop_rate)
        in_channels = out_channels
        # 4th block
        self.dense4 = DenseBlock(nblocks[3], in_channels, growth_rate, block, drop_rate)
        in_channels += nblocks[3]*growth_rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Linear(in_channels, num_classes)
        self.in_channels = in_channels
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(-1, self.in_channels)
        return self.fc(out)


def DenseNet121():
    return DenseNet([6, 12, 24, 16], bottleneck=True, growth_rate=32)


def DenseNet169():
    return DenseNet([6, 12, 32, 32], bottleneck=True, growth_rate=32)


def DenseNet201():
    return DenseNet([6, 12, 48, 32], bottleneck=True, growth_rate=48)


def DenseNet161():
    return DenseNet([6, 12, 24, 16], bottleneck=True, growth_rate=12)



