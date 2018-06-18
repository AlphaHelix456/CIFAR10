from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from models import densenet

def train(args, model, device, train_loader, criterion, optimizer, epoch):
    print('Train Epoch {}'.format(epoch))
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(output.data, 1)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total = (batch_idx+1) * len(data)
        if batch_idx % args.log_interval == 0:
            print('[{}/{}] ({:.0f}%)\tLoss: {:.4f} | Acc: {:.4f}%'.format(
                    total, len(train_loader.dataset), 100. * total / len(train_loader.dataset),
                    loss.item(), 100. * correct/total))

        state = {'state_dict': model.state_dict(), 'epoch': epoch}
        torch.save(state, './checkpoints/densenet121.t7')

def test(args, model, device, test_loader, criterion, epoch):
    print('Test Epoch {}'.format(epoch))
    running_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)	
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        _, pred = torch.max(output.data, 1)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Average Loss: {:.4f} | Acc: {:.4f}%'.format(
            running_loss/(args.test_batch_size), 100. * correct/len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, 
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=5, 
                        help='how many batches to wait before logging training status (default: 5)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed (default: 42)')
    parser.add_argument('--use-checkpoint', action='store_true', default=False, 
                        help='enables loading model from checkpoint')
    parser.add_argument('--no-cuda', action='store_true', default=False, 
                        help='disables CUDA training')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    start_epoch = 0 
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs) 	
	
    model = densenet.DenseNet121().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    if args.use_checkpoint:	
        checkpoint = torch.load('./checkpoints/densenet121')
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
	
    for epoch in range(start_epoch, args.epochs):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(args, model, device, test_loader, criterion, epoch)
	
if __name__ == '__main__':
    main() 
