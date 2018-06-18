import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=padding, bias=False)

def conv3x3(in_channels, out_channels, stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=padding, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, self.expansion*out_channels, stride=stride),
                nn.BatchNorm2d(self.expansion*out_channels)
                )
                            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut
        return F.relu(out)

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, self.expansion*out_channels)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, self.expansion*out_channels, stride=stride),
                nn.BatchNorm2d(self.expansion*out_channels)
                )               
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)
        
class ResNet(nn.Module):
    def __init__(self, block, nblocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = conv3x3(3, 64, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.res1 = self._make_layer(block, 64, nblocks[0], 1)
        self.res2 = self._make_layer(block, 128, nblocks[1], 2)
        self.res3 = self._make_layer(block, 256, nblocks[2], 2)
        self.res4 = self._make_layer(block, 512, nblocks[3], 2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channels, nblocks, stride):
        strides = [stride] + [1]*(nblocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.fc(out)

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(BottleneckBlock, [3, 4, 6, 3])

def ResNet101():
    return ResNet(BottleneckBlock, [3, 4, 23, 3])

def ResNet152():
    return ResNet(BottleneckBlock, [3, 8, 36, 3])
        
