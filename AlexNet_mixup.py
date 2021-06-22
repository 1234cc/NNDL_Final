# -*- coding: utf-8
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torchvision import transforms as transforms
import torch.backends.cudnn as cudnn


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

#定义swish
def swish(input):
    return input * torch.sigmoid(input)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, input):
        return swish(input)

#将数据转换为张量，并包装成Dataset类，用DataLoader来读取
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

#搭建卷积网络
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # Swish(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Swish(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Swish(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Swish(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Swish(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            # Swish(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # Swish(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        # print(x.shape)
        return x

use_cuda=torch.cuda.is_available()
if use_cuda:
        device = torch.device('cuda:0')
        cudnn.benchmark = False
else:
        device = torch.device('cpu')
net=AlexNet().to(device)
#设置训练参数
epoches = 40
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.001)

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#开始训练
total_loss = []
total_acc = []
for i in range(epoches):
    net.train()
    losses = []

    for train_x, train_y in loader:
        train_x, train_y=train_x.to(device), train_y.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(train_x, train_y,
                                                       1, True)

        outputs = net(inputs)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    total_loss.append(sum(losses)/len(losses))
    print("epoch {} loss : {}".format(i+1,sum(losses)/len(losses)))

    #计算在测试集上的ACC
    with torch.no_grad():
        net.eval()
        predict = net(torch.FloatTensor(test_data).to(device))
        predict = predict.cpu().detach().numpy()
        predict = np.argmax(predict, axis=1)
        acc = sum(predict == test_label) / len(predict)
        total_acc.append(acc)
        print('epoch {} test acc:{}'.format(i+1,acc))

fo = open("alex_batch_mixup.txt", "w")
fo.write('{}'.format(total_loss))
fo.write('\n')
fo.write('{}'.format(total_acc))

#
#保存训练好的模型
torch.save(net.state_dict(),"./saved_alex_net_mixup.pt")
#绘制loss曲线于acc曲线
import matplotlib.pyplot as plt
epo = np.arange(1,41)
plt.plot(epo,total_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss curve')
plt.savefig('./loss_curve_mixup.png')
plt.figure()#清空画板
plt.plot(epo,total_acc)
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('acc curve')
plt.savefig('./acc_curve_mixup.png')
