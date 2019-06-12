import torch
from torch import nn,optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

import net

# 然后可以定义一些超参数，如batch size 、learning rate 还有num_epoches等
# 超参数(Hyperparameters)
batch_size = 64
learning_rate = 1e-2
num_epoches = 20

# 接着需要进行数据预处理，就像之前介绍的，需要将数据标准化，这里运用到的函
# 数是torchvision.transforms. 它提供了很多图片预处理的方法。这里使用两个方
# 法:第1个是transforms.ToTensor() .第二个是transforms.Normalize()
# transform.ToTensor() 很好理解.就是将图片转换成PyTorch 中处理的对象Ten­
# sor.在转化的过程中PyTorch 自动将图片标准化了，也就是说Tensor 的范用是O~ 1
# 接着我们使用transforms.Normalize() . 需要传入两个参数:第一个参数是均值，第
# 二个参数是方差，做的处理就是减均值,再除以方差
data_tf = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])]
)

# 这里transforms.Compose() 将各种预处理操作组合到一起，transforms.Normalize([0.5], [0.5]) 
# 表示减去0.5 再除以0.5 ，这样将图片转化到了-1~1 之间，
# 注意因为图片是灰度图，所以只有一个通道，如果是彩色的图片，有三通道，那么
# 用transforms .Normalize ([a, b , c] , [d, e , f]) 来表示每个通道对应的均值和方差

# 然后读取数据集
# 下载训练集MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root = './data',train = True ,transform=data_tf,download=True)

test_dataset  = datasets.MNIST(root='./data',train=False,transform=data_tf)

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# 通过PyTorch 的内置函数torchvision.datasets.MNIST 导人数据集，传入数
# 据预处理，前面介绍了如何定义自己的数据集，之后会用具体的例子说明。接着使用
# torch.utils.data.DataLoader 建立一个数据迭代器，传入数据集和batch size,
# 通过shuffle=True 来表示每次迭代数据的时候是否将数据打乱。

# 接着导入网络，定义损害函数和优化方法
# model = net.simpleNet(28*28,300,100,10)
# model = net.Activation_Net(28*28,300,100,10)
# model = net.Batch_Net(28*28,300,100,10)

model = net.SimpleCNN()
print(model)
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = learning_rate)

# net.simpleNet 是定义的简单三层网络，里面的参数是28 x 28, 300, 100, 10 ,其中
# 输入的维度是28 x 28，因为输入图片大小是28 x 28 ，然后定义两个隐藏层分别是300
# 和100 ，最后输出的结果必须是10 ，因为这是一个分类问题，二共有0 ~9 这10 个数
# 字，所以是10 分类。损失函数定义为分类问题中最常见的损失函数交叉熵，使用随机
# 梯度下降来优化损失函数。
# 接着开始训练网络，流程基本和之前二致，这里就不再赘述士最后训练完网络之后
# 需要测试网络，通过下面的代码来测试
model.eval()
eval_loss = 0
eval_acc = 0
for data in test_loader:
    img, label = data
    img = img.view(img.size(0),-1)
    if torch.cuda.is_available():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    else:
        img = Variable(img)
        label = Variable(label)
    out = model(img)
    loss = criterion(out,label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = torch.sum(pred == label)
    eval_acc += num_correct.item()
print('Test Loss:{:.6f},Acc:{:.6f}'.format(
    eval_loss / (len(test_dataset)),
    eval_acc / (len(test_dataset))))



