import torch
from torch.autograd import Variable
# Create Variable
# 构建飞1ariable. 要谊意得传入一个参数requires_grad=True ，这个参数表示是否
# 对这个变量求梯度，默认的是False ，也就是不对这个变量求梯度，这里我们希望得到
# 这些变量的梯度，所以需要传入这个参数c

x = Variable(torch.Tensor([1]),requires_grad=True)
w = Variable(torch.Tensor([2]),requires_grad=True)
f = Variable(torch.Tensor([3]),requires_grad=True)

# Build a computational graph
y = w * x + f # y = 2 * x + 3
# Compute gradients
y.backward() #sanme as y.backward(torch.FloatTensor([1]))
# Print out the gradients
print(x.grad) # x.grad = 2
print(w.grad) # w.grad = 2
print(f.grad) # f.grad = 2

# 从上面的代码中，我们注意到了一行y.backward() ，这一行代码就是所谓的自动
# 求导，这其实等价于y.backward(torch.FloatTensor([ 工J ) ) ，只不过对于标量求导
# 里面的参数就可以不写了，自动求导不需要你再去明确地写明哪个函数对哪个函数求
# 导，直接通过这行代码就能对所有的需要梯度的变量进行求导，得到它们的梯度，然后
# 通过x.grad 可以得到x 的梯度=
# 上面是标量的求导，同时也可以做矩阵求导，比如:
x = torch.randn(3)
x = Variable(x,requires_grad=True)

y = x*2
print(y)

y.backward(torch.FloatTensor([1,0.1,0.01]))
print(x.grad)

# 相当于给出了一个兰维向量去做运算，这时候得到的结果ν就是一个向量，这里
# 对这个向量求导就不能直接写成y.backward() ，这样程序是会报错的。这个时候需
# 要传入参数声明，比如y.backward(torch.FloatTensor 门工，工，1) )) 唔这样得到的
# 结果就是它们每个分量的梯度，或者可以传入y.backward(torch.FloatTensor( [工，
# 0.1 ，0. 0 1] )) ，这样得到的梯度就是它们原本的梯度分别乘上1 ，0.1 和0.01 。
