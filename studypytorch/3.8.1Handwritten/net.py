import torch
from torch import nn

# 在PyTorch 里面可以很简单地走义三层全连接神经网络。

class simpleNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,out_dim)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
# 对于这个三层网络，需要传递进去的参数包括:输入的维度、第一层网络的神经元
# 个数、第二层网络神经元的个数，以及第气层网络(输出层)神经元的个数2

# 接着改进下网络，添加激活雨数增加网络的非线性，方法也非常简单二
class Activation_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Activation_Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
# 这里只需要在每层网络的输出部分添加激活函数就可以了，用到nn.Sequential()，
# 这个函数是将网络的层组合到一起，比如上面将nn.Linear ()和nn.ReLU()组合到
# 一起作为self.layer : 注意最后一层输出层不能添加激活函数，因为输出的结果表示
# 的是实际的得分.

# 最后添加-个加快收敛速度的方法一一批标准化c
class Batch_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Batch_Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
# 同样使用nn.Sequential( )将nn. BatchNormld ()组合到网络层中，注意.m标
# 准化-般放在全连接层的后面、非线性层(激活函数)的前面

# 简单多层卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN,self).__init__() # b, 3, 32, 32
        layer1 = nn.Sequential()
        layer1.add_module('conv1', nn.Conv2d(3, 32, 3, 1, padding=1)) # b, 32, 32, 32
        layer1.add_module('relu1', nn.ReLU(True))
        layer1.add_module('pool1', nn.MaxPool2d(2, 2)) # b, 32, 16, 16
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2', nn.Conv2d(32, 64, 3, 1, padding=1)) # b, 64, 16, 16
        layer2.add_module('relu2', nn.ReLU(True))
        layer2.add_module('pool2', nn.MaxPool2d(2, 2)) # b, 64, 8, 8
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module('conv3', nn.Conv2d(64, 128, 3, 1, padding=1)) # b, 128, 8, 8
        layer3.add_module('relu3', nn.ReLU(True))
        layer3.add_module('pool3', nn.MaxPool2d(2, 2)) # b, 128, 4, 4
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1', nn.Linear(2024, 512))
        layer4.add_module('fc_relu1', nn.ReLU(True))
        layer4.add_module('fc2', nn.Linear(512, 64))
        layer4.add_module('fc_relu2', nn.ReLU(True))
        layer4.add_module('fc3', nn.Linear(64, 10))
        self.layer4 = layer4

        def forward(self, x):
            conv1 = self.layer1(x)
            conv2 = self.layer2(conv1)
            conv3 = self.layer3(conv2)
            fc_input = conv3.view(conv3,size(0),-1)
            fc_out = self.layer4(fc_input)
            return fc_out

# 多层卷积神经网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1) , # b,16,26,26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1), #b,32,24,24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2) #b,32,12,12
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1), # b,64,10,10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1), # b ,128,8,8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2) # b,128,4,4
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4 ,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10)
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
