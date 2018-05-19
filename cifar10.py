from __future__ import print_function

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from models import densenet

use_cuda = torch.cuda.is_available()
use_checkpoint = False
best_accuracy = 0
epochs = 150
batch_size = 64

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck']

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)

net = densenet.DenseNet121()

if use_checkpoint:
    checkpoint = torch.load('./checkpoints/densenet121')
    net.load_state_dict(checkpoint['state_dict'])
    best_accuracy = checkpoint['accuracy']
    start_epoch = checkpoint['epoch']


if use_cuda:
    net = nn.DataParallel(net)

if use_cuda:
    net.cuda()

    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)


def train(epoch):
    print('Epoch %d/%d' % (epoch, epochs))
    global best_accuracy
    running_loss = 0
    total = 0
    correct = 0
    for i, (inputs, labels) in enumerate(trainloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
            
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += batch_size
        correct += predicted.eq(labels.data).cpu().sum()
        print ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (running_loss/(i+1), 100*correct/total, correct, total))

    scheduler.step()
    accuracy = 100*correct/total
    if accuracy > best_accuracy:
        state = {'state_dict': net.state_dict(),
                 'acc': accuracy,
                 'epoch': epoch,
                 }
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
        torch.save(state, './checkpoints/densenet121.t7')
        best_accuracy = accuracy

def test():
    running_loss = 0
    total = 0
    correct = 0
    for i, (inputs, labels) in enumerate(testloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
            
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += batch_size
        correct += predicted.eq(labels.data).cpu().sum()

    print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
          % (running_loss/(i+1), 100*correct/total, correct, total))


if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
    test()
    
