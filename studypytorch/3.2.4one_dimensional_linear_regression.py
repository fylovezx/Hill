import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

x_train = np.array([[3.3],[4.4],[5.5],[6.71],[6.93],[4.168],
[9.779],[6.182],[7.59],[2.167],[7.042],
[10.791],[5.313],[7.997],[3.1]],dtype=np.float32)

y_train = np.array([[1.7],[2.76],[2.09],[3.19],[1.694],[1.573],
[3.366],[2.596],[2.53],[1.221],[2.827],
[3.465],[1.65],[2.904],[1.3]],dtype=np.float32)

fig, ax = plt.subplots()
ax.scatter(x_train, y_train)

ax.set(xlabel='x_train', ylabel='y_train',
       title='x_train,y_train scatter Image')
ax.grid()
plt.show()

# 我们想要做的事情就是找一条直线去逼近这些点，也就是希望这条直线离这些点
# 的距离之和最小，先将numpy.array 转换成Tensor ，因为PyTorch 里面的处理单元是
# Tensor ，按上←章讲的方法，这就特别简单了:
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
# 接着需要建立模型，根据上一节PyTorch 的基础知识，这样来定义一个简单的模型:
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.linear = nn.Linear(1,1) # input and output is 1 dimension
#     nn.Linear( in，out) ,
# 它就是在PyTorch 里用来表示一个全连接神经网络层的函数，比如输入层4 个节点，输
# 出2 个节点，可以用nn.Linear(4 ，2) 来表示
    def forward(self, x):
        out = self.linear(x)
        return out
    
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

# 这里我们就定义了一个超级简单的模型ν= wx + b ，输入参数是一维，输出参数
# 也是一维，这就是一条直线，当然这里可以根据你想要的输入输出维度进行更改，我们
# 希望去优化参数w 和b 能够使得这条直线尽可能接近这些点，如果支持GPU 加速，可
# 以通过model. cuda ()将模型放到GPU 上。
# 然后定义损失函数和优化函数，这里使用均方误差作为优化函数，使用梯度下降
# 进行优化:
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 1e-3)
# 接着就可以开始训练我们的模型了:
num_epochs = 1000 # 步骤1.定义好我们要跑的epoch 个数，
for epoch in range(num_epochs):
    if torch.cuda.is_available():
        inputs = Variable(x_train).cuda()
        target = Variable(y_train).cuda()
    else:
        inputs = Variable(x_train)
        target = Variable(y_train)
    # 步骤2.数据变成Variable 放入计算图
    #forward
    out = model(inputs) # 步骤3.通过out=model(inputs) 得到网络前向传播得到的结果
    loss = criterion(out,target) # 步骤4.通过loss=criterion(out,target)得到损失函数
    # backward
    optimizer.zero_grad() # 步骤5.然后归零梯度，做反向传播和更新参数
    loss.backward()
    optimizer.step()

    # 在训练的过程中隔一段时间就将损失函数的值打印出米看看，确保我们的模型误差越来越小.
    if(epoch+1) % 20 ==0:
        # 注意loss.data[O] 不能用了修改为item()
        print('Epoch[{}/{},loss:{:.6f}'.format(epoch+1,num_epochs,loss.item()))


# 定义好我们要跑的epoch 个数，然后将数据变成Variable 放入计算图，然后通
# 过out=model(inputs) 得到网络前向传播得到的结果，通过loss=criterion(out,target)
# 得到损失函数，然后归零梯度，做反向传播和更新参数，特别要注意的是，每
# 次做反向传播之前都要归零梯度，optimizer.zero_grad().不然梯度会累加在一起，
# 造成结果不收敛.在训练的过程中隔一段时间就将损失函数的值打印出米看看，确保
# 我们的模型误差越来越小.注意loss.data[O] ，首先loss 是一个Variable ，所以通过
# loss.data 可以取出-个Tensor ，再通过loss.data[O] 得到二个int 或者float 类型
# 的数据，这样我们才能够打印出相应的数据已
# 做完训练之后可以预测一下结果.

model.eval()

predict = model(inputs) #inputs = Variable(x_train).cuda()
if torch.cuda.is_available():
    predict = predict.data.cpu().numpy()
else:
    predict = predict.data.numpy()

plt.plot(x_train.numpy(),y_train.numpy(),'ro',label='Original data')
plt.plot(x_train.numpy(),predict,label='Fitting Line')
plt.show()
# 首先需要通过model . eval ()将模型变成测试模式，这是因为有一些层操作，比
# 如Dropout 和BatchNormalization 在训练和测试的时候是不一样的，所以我们需要通过
# 这样一个操作来转换这些不一样的层操作.然后将测试数据放入网络做前向传播得到
# 结果
