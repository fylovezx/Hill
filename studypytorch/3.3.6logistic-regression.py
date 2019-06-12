import os
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 我们先从data. txt 文件中读取数据
with open('studypytorch\\data\\3.3.6data.txt','r') as f:
    data_list = f.readlines()
    data_list = [i.split('\n')[0] for i in data_list]
    data_list = [i.split(',') for i in data_list]
    data = [(float(i[0]),float(i[1]),float(i[2])) for i in data_list]

# 标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0]/x0_max, i[1]/x1_max, i[2]) for i in data]

x0 = list(filter(lambda x: x[-1] == 0.0, data)) # 选择第一类的点
x1 = list(filter(lambda x: x[-1] == 1.0, data)) # 选择第二类的点

plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]
# 然后通过matplotlib 能够简单地将数据画出来士
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')
# plt.show()
# 接下来我们将数据转换成 NumPy 的类型，接着转换到 Tensor 为之后的训练做准备
np_data = np.array(data, dtype='float32') # 转换成 numpy array
x_data = torch.from_numpy(np_data[:, 0:2]) # 转换成 Tensor, 大小是 [100, 2]
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1) # 转换成 Tensor，大小是 [100, 1]

# 接下来定义Logistic 回归的模型，以及二分类问题的损失函数和优化占法二
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr = nn.Linear(2,1)
        self.sm = nn.Sigmoid()
    def forward(self,x):
        x = self.lr(x)
        x = self.sm(x)
        return x

logistic_model = LogisticRegression()
if torch.cuda.is_available():
    logistic_model.cuda()

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(logistic_model.parameters(),lr = 1e-3,momentum=0.9)

# 这里nn.BCELoss 是二分类的损失雨数，torch.optim.SGD 是随机梯度下降优化函数。
# 然后训练模型，并且间隔一定的迭代次数输出结果

for epoch in range(20000) :
    if torch.cuda.is_available():
        x = Variable(x_data).cuda()
        y = Variable(y_data).cuda()
    else:
        x = Variable(x_data)
        y = Variable(y_data)
    # forward
    out = logistic_model(x)
    loss = criterion(out,y)
    print_loss = loss.item()
    mask = out.ge(0.5).float()
    correct = (mask ==y ).sum()
    acc = correct.item() /x.size(0)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(epoch + 1) % 10000 == 0 :
        print('*'*10)
        print('epoch{}'.format(epoch+1))
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:4f}'.format(acc))
# 其中mask=out.ge(O.5) .float() 是判断输出结果如果大于0.5 就等于1 ，小于
# 0.5 就等于0 ，通过这个来计算模型分类的准确率
# 我们可以将这条直线画出来，因为模型中学习的参数叫，问和b 其实构成了一条
# 直线ωlX 十W2Y 十b = 0 ，在直线上方是一类，在直线下方又是一类.我们可以通过下
# 面的方式将模型的参数取出来，并将直线画出来
w0,w1 = logistic_model.lr.weight[0]
w0 = w0.item()
w1 = w1.item()
b = logistic_model.lr.bias.item()
# plot_x = np.arange(30,10,0.1)
plot_x = np.arange(0.2, 1, 0.01)
plot_y = (-w0 * plot_x-b) /w1
plt.plot(plot_x,plot_y)
plt.show()