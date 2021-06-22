'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset,DataLoader,TensorDataset

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import csv
import pickle
from resnet import *
from utils import progress_bar, mixup_data, mixup_criterion
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='mixup_default', type=str, help='session id')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--alpha', default=1., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
args = parser.parse_args()

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
base_learning_rate = 0.1
if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu

# Data
print('==> Preparing data..')
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

labels = []
train_batch = []

#读取并合并5个batch的训练数据
for i in range(5):
    data_batch = unpickle('./cifar-10-batches-py/data_batch_'+str(i+1))
    labels.extend(data_batch[b'labels'])
    train_batch.append(data_batch[b'data'].reshape(10000,3,32,32))

train_data = np.vstack(batch for batch in train_batch) #(50000,3,32,32)
labels = np.array(labels)#(5000,)
class LaneDataSet(Dataset):
    def __init__(self, x, y,transform):
        self.x_list = x
        self.y_list = y
        self.transform=transform
        assert len(self.x_list) == len(self.y_list)
        # pass  # 省略了

    def __len__(self):
         return len(self.x_list)

    def __getitem__(self, idx):
        x_one = self.x_list[idx]
        y_one = self.y_list[idx]
        if self.transform:
            t=transforms.Compose([ transforms.RandomHorizontalFlip(),transforms.ToTensor()])
            x=t(x_one)
            return (x,y_one)
        else:
            return (x_one, y_one)
# print(type(train_data))
x = torch.FloatTensor(train_data)
y = torch.LongTensor(labels).long()

dataset = LaneDataSet(x,y,transform=False)
loader = DataLoader(dataset=dataset,batch_size=64,shuffle=True)

#导入测试集
test = unpickle('./cifar-10-batches-py/test_batch')
test_label = np.array(test[b'labels'])
test_data = test[b'data'].reshape(10000,3,32,32)
test_dataset = LaneDataSet(test_data ,test_label,transform=False)
test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=False)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = PreActResNet18()
    net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + '_' + args.sess + '_' + str(args.seed) + '.csv'

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx,(inputs,targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)
        optimizer.zero_grad()
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        outputs = net(inputs)

        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return (train_loss/batch_idx, 100.*correct/total)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    # for batch_idx, (inputs, targets) in enumerate(test_data):
    for batch_idx, (inputs, targets) in enumerate(test_loader):
    # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        inputs, targets = torch.FloatTensor(inputs), torch.LongTensor(labels).long()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss/batch_idx, 100.*correct/total)

def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed))


if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(100):
    train_loss, train_acc = train(epoch)
    print('epoch {} train loss:{}'.format(epoch + 1, train_loss))
    test_loss, test_acc = test(epoch)
    print('epoch {} test acc:{}'.format(epoch + 1, test_acc))
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
