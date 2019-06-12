# 在处理任何机器学习问题之前都需要数据读取，并进行预处理。PyTorch 提供了很
# 多工具使得数据的读取和预处理变得很容易
# torch.utils.data.Dataset 是代表这一数据的抽象类，你可以自己定义你的数
# 据类继承和重写这个抽象类，非常简单，只需要定义len一和_getitem一这两个函数:
class myDataset(Dataset):
    def __init__(self,csv_file,txt_file,root_dir,other_file):
        self.csv_data = pd.read_csv(csv_file)
        with open(txt_file,'r') as f:
            data_list = f.readlines()
        self.txt_data = data_list
        self.root_dir = root_dir
    
    def __len__(self):
        return len(self.csv_data)

    def __getitem(self,idx):
        data = (self.csv_data[idx],self.txt_data[idx])
        return data
# 通过上面的方式，可以定义我们需要的数据类，可以通过迭代的方式来取得每一
# 个数据，但是这样很难实现取batch ，shuffle 或者是多线程去读取数据，所以PyTorch
# 中提供了一个简单的办法来做这个事情，通过torch.utils.data.DataLoader 来定
# 义一个新的迭代器.如下:

dataiter = DataLoader(myDataset,batch_size=32,shuffle=True,collate_fn=default_collate)
# 里面的参数都特别清楚，只有collate fn 是表示如何取样本的，我们可以定义
# 自己的函数来准确地实现想要的功能，默认的函数在一般情况下都是可以使用的.
# 另外在torchvision 这个包中还有二个更高级的有关于计算机视觉的数据读取类:
# ImageFolder ，主要功能是处理图片，且要求图片是下面这种存放形式:
root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png
root/cat/123.png
root/cat/asd .png
root/cat/zxc.png
# 之后这样来调用这个类:
dset = ImageFolder(root='root_path',transform=None,loader=default_loader)
# 其中的root 需要是根目录，在这个目录下有几个文件夹，每个文件夹表示一个类
# 别: transform 和target....transform 是图片增强，之后我们会详细讲; loader 是图片
# 读取的办法，因为我们读取的是图片的名字，然后通过loader 将图片转换成我们需要
# 的图片类型进入神经网络c
