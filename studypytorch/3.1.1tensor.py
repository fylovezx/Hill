import torch,numpy as np
# 定义一个三行两列给定元素的矩阵，并显示矩阵的元素和大小
a = torch.Tensor([[2,3],[4,8],[7,9]])
print('a is :{}'.format(a))
print('a size is {}'.format(a.size()))

# 需要注意的是torch.Tensor 默认的是torch.FloatTensor 数据类型，也可以
# 定义我们想要的数据类型，就像下面这样:
b = torch.IntTensor([[2,3],[4,8],[7,9]])
print('b is :{}'.format(b))
# 当然也可以创建一个全是O 的空Tensor 
c = torch.zeros((3,2))
print('zeros tensor :{}'.format(c))
# 或者取一个正态分布作为随机初始值:
d = torch.randn((3,2))
print('normal randon is :{}'.format(d))

# 除此之外，还可以在Tensor 与numpy.ndarray 之间相互转换;
numpy_b = b.numpy()
print('conver to numpy is \n {}'.format(numpy_b))

e = np.array([[2,3],[4,5]])
torch_e = torch.from_numpy(e)
print('from numpy to torch.Tensor is {}'.format(torch_e))
f_torch_e = torch_e.float()
print('change data type to float tensor:{}',format(f_torch_e))
# 通过简单的b. numpy () ，就能将b 转换为numpy 数据类型，同时使用torch.from
# num py ()就能将numpy 转换为tensor ，如果需要更改tensor 的数据类型，只需要在转
# 换后的tensor 后面加上你需要的类型.比如想将a 的类型转换成f1oat ，只需a. float ()
# 就可以了。


# 如果你的电脑支持GPU 加速，还可以将Tensor 放到GPU 上。
# 首先通过torch.cuda.is available() 判断一下是否支持GPU ，如果想把tensor
# a 放到GPU 上，只需a. cuda ()就能够将tensor a 放到Ij GPU 上了。
print(torch.cuda.is_available())
if torch.cuda.is_available():
    a_cuda = a.cuda()
    print(a_cuda)

