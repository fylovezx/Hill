import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 首先需要预处理数据，也就是需要将数据变成一个矩阵的形式:
# 在PyTorch 里面使用torch. cat ()函数来实现Tensor 的拼接:

def make_features(x):
    # Builds features i.e a matrix with columns [x,x^2,x^3].
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1,4)],1)

# 对于输入的n 个数据，我们将其扩展成上面矩阵所示的样子仨
# 然后定义好真实的函数:
W_target = torch.FloatTensor([1,-1,1]).unsqueeze(1)
b_target = torch.FloatTensor([1])


def f(x):
    # Approximated function.
    return x.mm(W_target) + b_target[0]

# 这里的权重已经定义好了，unsqueeze(l ) 是将原来的tensor 大小由3 变成(3 ，1),
# x.mm(W target ) 表示做矩阵乘法，f( x ) 就是每次输入一个x得到一个y的真实函数。

# 在进行训练的时候我们需要采样一些点，可以随机生成一些数来得到每次的训
# 练集:

def get_batch(batch_size=10):
    # Builds a batch i.e. (x,f(x)) pair.
    # torch.randn返回一个张量，从标准正态分布（均值为0，方差为1）中抽取的一组随机数。张量的形状由参数sizes定义。
    random = torch.randn(batch_size)
    random = np.sort(random)
    random = torch.Tensor(random)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)

# 通过上面这个函数我们每次取batch sìze 这么多个数据点，然后将其转换成矩
# 阵的形式，再把这个值通过函数之后的结果也返回作为真实的目标。

# 然后可以定义多项式模型:
class poly_model(nn.Module):
    def __init__(self):
        # super() 函数是用于调用父类(超类)的一个方法
        super(poly_model,self).__init__()
        self.poly = nn.Linear(3,1)

    def forward(self,x):
        out = self.poly(x)
        return out

if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

# 这里的模型输入是3 维，输山是1 维，j~之前定义的线性模型只有很小的差别。
# 然后我们定义损失函数和优化器:

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 1e-3)

# 同样使用均方误差来衡量模型的好坏，使用随机梯度下降来优化模型，然后开始
# 训练模型:

epoch = 0
while True:
    # Get data
    batch_x,batch_y = get_batch()
    # Forward pass
    out = model(batch_x)
    loss = criterion(out,batch_y)
    # 注意loss.data[O] 不能用了修改为item()
    print_loss = loss.item()
    # Reset gradients
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # update parameters
    optimizer.step()
    epoch += 1
    if print_loss < 1e-3:
        break

print("Loss: {:.6f}  after {} batches".format(loss.item(), epoch))

print(
    "==> Learned function: y = {:.2f} + {:.2f}*x + {:.2f}*x^2 + {:.2f}*x^3".format(model.poly.bias[0], model.poly.weight[0][0],
                                                                                    model.poly.weight[0][1],
                                                                                    model.poly.weight[0][2]))
print("==> Actual function: y = {:.2f} + {:.2f}*x + {:.2f}*x^2 + {:.2f}*x^3".format(b_target[0], W_target[0][0],
                                                                                    W_target[1][0], W_target[2][0]))

model.eval() # 加与不加的区别？
predict = model(batch_x)

# batch_x,batch_y = get_batch() 这个地方为什么不行重新建一次
batch_x = batch_x.cpu()
batch_y = batch_y.cpu()
x = batch_x.numpy()[:, 0]
# print(batch_x.numpy())
plt.plot(batch_x.numpy()[:, 0], batch_y.numpy(), 'bo')
plt.plot(batch_x.numpy()[:, 1], batch_y.numpy(), 'ro')
plt.plot(batch_x.numpy()[:, 2], batch_y.numpy(), 'go')
predict = predict.cpu()
predict = predict.data.numpy()

plt.plot(x, predict, 'b')
plt.plot(batch_x.numpy()[:, 1], predict, 'r')
plt.plot(batch_x.numpy()[:, 2], predict, 'g')
plt.show()
a =1



# 这里我们希望模型能够不断地优化，直到实现我们设立的条件，取出的32 个点的
# 均方误差能够小于0.001 。


