# 深度学习

## 教程

+ [DEEP LEARNING WITH PYTORCH: A 60 MINUTE BLITZ](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)：PyTorch官方教程
+ [Dive-into-DL-PyTorch](https://github.com/ShusenTang/Dive-into-DL-PyTorch/)：《动手学深度学习》PyTorch版本，含电子版和torch代码。原书中代码框架为MXNet，先进但不够流行
+ [神经网络与深度学习](https://nndl.github.io/nndl-book.pdf)
+ [paddlepaddle飞桨](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/index_cn.html)：paddlepaddle官方文档

深度学习和神经网络并不等同，深度学习可以使用神经网络，也可以使用深度信念网络等概率图模型。

## PaddlePaddle

### 静态图示例（声明式编程）

+ 不能使用python提供的while、if等控制流。定义执行器，在执行该执行器并传入数据时，才真正运行模型。操作会放入program中，通过Executor进行编译，然后执行Executor。通常静态图更快，因为不需要小粒度op进行python与cpp交互

```python
# 张量相加
import paddle.fluid as fluid
import numpy as np

# 定义tensor
a = fluid.data(name='a', shape=[None, 1], dtype='int64')
b = fluid.data(name='b', shape=[None, 1], dtype='int64')

# 定义模型
result = fluid.layers.elementwise_add(a, b)

# 准备运行
place = fluid.CPUPlace()  # 定义运算设备
exe = fluid.Executor(place)  # 创建执行器
exe.run(fluid.default_startup_program())  # 网络参数初始化，并为变量分配空间

# 输入数据
data1 = np.array([1, 2, 3]).reshape(-1, 1)
data2 = np.array([1, 2, 3]).reshape(-1, 1)

# 运行网络
output = exe.run(
    feed={'a': data1, 'b': data2},
    fetch_list=[a, b, result]  # 需要打印的变量list
	)
print(output)
```

```python
import paddle
import paddle.fluid as fluid
import numpy as np
from paddle.utils.plot import Ploter
import math
import matplotlib
import matplotlib.pyplot as plt


BATCH_SIZE = 20
train_reader = fluid.io.batch(fluid.io.shuffle(paddle.dataset.uci_housing.train(), buf_size=500), batch_size=BATCH_SIZE)
test_reader = fluid.io.batch(fluid.io.shuffle(paddle.dataset.uci_housing.test(), buf_size=500), batch_size=BATCH_SIZE)

x = fluid.data(name='x', shape=[None, 13], dtype='float32')
y = fluid.data(name='y', shape=[None, 1], dtype='float32')
y_predict = fluid.layers.fc(input=x, size=1, act=None, bias_attr=True)

main_program = fluid.default_main_program()  # 获取默认/全局主函数
startup_program = fluid.default_startup_program()  # 获取默认/全局启动程序

cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_loss = fluid.layers.mean(cost)

test_program = main_program.clone(for_test=True)  # 克隆main_program得到test_program
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)

use_cuda = False
place = fluid.CPUPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)

num_epochs = 100

def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program, feed=feeder.feed(data_test), fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]
        count += 1
    return [x_d / count for x_d in accumulated]

params_dirname = "fit_a_line.inference.model"
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe.run(startup_program)
train_prompt = "train cost"
test_prompt = "test cost"
plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

exe_test = fluid.Executor(place)

for pass_id in range(num_epochs):
    for data_train in train_reader():
        avg_loss_value, =exe.run(main_program, feed=feeder.feed(data_train), fetch_list=[avg_loss])
        if step % 10 == 0:
            plot_prompt.append(train_prompt, step, avg_loss_value[0])
            plot_prompt.plot()
            print("%s, Step %d, Cost %f" % (train_prompt, step, avg_loss_value[0]))
        if step % 100 == 0:
            test_metics = train_test(executor=exe_test, program=test_program, reader=test_reader, fetch_list=[avg_loss.name], feeder=feeder)
            plot_prompt.append(test_prompt, step, test_metics[0])
            plot_prompt.plot()
            print("%s, Step %d, Cost %f" % (test_prompt, step, test_metics[0]))
            if test_metics[0] < 10.0: # 如果准确率达到要求，则停止训练
                break
    step += 1

    if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")
    
    if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()
def save_result(points1, points2):
    matplotlib.use('Agg')
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()

    plt.savefig('./image/prediction_gt.png')

with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe) # 载入预训练模型
    batch_size = 10

    infer_reader = fluid.io.batch(paddle.dataset.uci_housing.test(), batch_size=batch_size) # 准备测试集

    infer_data = next(infer_reader())
    infer_feat = np.array([data[0] for data in infer_data]).astype("float32") # 提取测试集中的数据
    infer_label = np.array([data[1] for data in infer_data]).astype("float32") # 提取测试集中的标签

    assert feed_target_names[0] == 'x'
    results = infer_exe.run(inference_program, feed={feed_target_names[0]: np.array(infer_feat)}, fetch_list=fetch_targets) # 进行预测
    #打印预测结果和标签并可视化结果
    print("infer results: (House Price)")
    for idx, val in enumerate(results[0]):
        print("%d: %.2f" % (idx, val)) # 打印预测结果

    print("\nground truth:")
    for idx, val in enumerate(infer_label):
        print("%d: %.2f" % (idx, val)) # 打印标签值

    save_result(results[0], infer_label)
```

### 动态图示例（命令式编程）

+ 使用`with fluid.dygraph.guard():`开启动态图模式

```python
import paddle.fluid as fluid
import numpy as np
from fluid.dygraph.base import to_variable


data = np.ones([2, 2], np.float32)

# 开启动态图模式
with fluid.dygraph.guard():
    x = to_variable(data)
    x += 10
    print(x.numpy())
```



### 算子

+ `fluid.layers`或`fluid.nets`中提供
    + `fluid.layers.fill_constant(shape=[3, 2], value=16, dtype="int64")`：值都为16的3行2列张量
    + `fluid.layers.Print(x, message="Print x:")`：模型中==打印==张量x
    + `fluid.layers.while_loop(cond, body, loop_vars)`：静态图中的while，在cond函数为真时执行body函数，两函数都传入loop_vars变量
    + `fluid.layers.create_parameter(name, shape, dtype)`：创建==可学习的variable==
    + `fluid.layers.fc(input, size)`：全连接层

## PyTorch

### 检查版本

```python
# python下
torch.__version__               # PyTorch版本
torch.version.cuda              # CUDA版本
torch.backends.cudnn.version()  # cuDNN版本
torch.cuda.get_device_name(0)   # GPU类型
torch.cuda.is_available()		# GPU是否可用
torch.cuda.device_count()		# 本次任务已申请的GPU数量
# bash下
nvidia-smi						# 查看GPU显卡详情
```

### 运行代码

`CUDA_VISIBLE_DEVICES=0,1 python train.py`指定GPU型号，或在代码里`os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'`

### 引用包

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as opt
```

### cuDNN

cuDNN是深度学习在GPU的加速库，在底层优化了卷积、池化等操作，比直接使用GPU快。Pytorch默认使用cnDNN加速，但`torch.backends.cudnn.benchmark`默认为False。如果手动设置为True，可以进一步==加速模型的卷积层==。

原因是`torch.backends.cudnn.benchmark=True`会使训练前选择最合适的卷积实现算法，即在训练前花时间对每层进行卷积算法测试和选择，这就要求模型的架构、层的输入输出维度不变，否则耗时更长。不过一般的模型结构都不会动态变化。

该设置一般在model.cuda()后。

### 固定随机数种子

```python
np.random.seed(SEED)  # numpy种子
torch.manual_seed(seed)  # torch种子
torch.cuda.manual_seed_all(seed)  # cuda种子

# 同时模型里不能使用下面这句话，不使用cnDNN加速
torch.backends.cudnn.benchmark = True
# 同时模型里需要使用这句话，保证卷积算法的选择固定
torch.backends.cudnn.deterministic = True
```

随机数种子固定后，模型每次结果相同，以复现结果。注意随机数种子固定不会影响dropout和bn，甚至数据切分可能随机，因此在model.train()模式下依然会使得结果变动。

### 判断cuda可用

`torch.cuda.is_available()`

### tensor

tensor和np.ndarray类似，但可以放在GPU上。

#### 查询

+ `x.dtype`：查看张量x内部数据的类型
+ `x.size()`：查看张量x的维度
+ `assert tensor.size() == (N, D, H, W)`：模型运行中检查维度符合预期
+ `tensor.is_cuda`：查询是否为GPU张量
+ `tensor.device`：查询tensor位置

#### 创建

+ `torch.empty(x, y)`：创建维度x\*y的未初始化张量，不一定为全0
+ `torch.rand(x, y)`：创建维度x\*y的随机张量，范围是0-1
+ `torch.randn(x, y)`：创建维度x\*y的随机且正态分布张量，均值0方差1
+ `torch.zeros(x, y)`：创建维度x\*y的全0向量
+ `torch.ones(x, y)`：创建维度x\*y的全1向量
+ `torch.full([x, y], n)`：创建维度x\*y的值全为n的张量
+ `torch.eye(x)`：创建维度x\*x的单位张量
+ `torch.arange(start, end, step)`：创建一维等差数列张量
+ `torch.xxx_like(tensor1)`：创建和tensor1维度相同的xxx类型向量，如ones_like等

#### 转换

+ `torch.tensor([1, 2, 3])`：从list或array直接变为tensor，==不共享内存==
+ `torch.from_numpy(array)`：从array变为tensor，==共享内存==
+ `tensor1.numpy()`：从cpu tensor转为numpy，==共享内存==
+ `tensor1.view(x, y)`：对tensor1进行==维度修改==，其中维度-1表示任意，先分配其他维，==共享内存==。不能处理经过transpose操作的内存不连续的张量。推荐使用`tensor1.clone().view(x, y)`，clone操作保证了副本不共享内存，但共享梯度
+ `tensor1.reshape((x, y))`：和view相同，可以处理内存中不连续的张量，如经过transpose操作的张量。但reshape返回的结果==可能共享内存也可能不共享==，因此不推荐使用
+ `tensor1.item()`：从1维张量中得到值
+ `tensor1 = tensor1.cuda()`：将张量放到GPU
+ `tensor1 = tensor1.cpu()`：将张量放到CPU

#### 计算

+ `tensor1 + tensor2`或`torch.add(tensor1, tensor2)`或`tensor1.add_(tensor2)`：维度相同时，按元素相加。维度不同时，一个1\*m(或m\*1)张量和n\*1(1\*n)相加，得到n\*m的元素相加的结果，即PyTorch的==broadcasting机制==，先复制后计算
+ `torch.mm(tensor1, tensor2)`：矩阵相乘
+ `torch.bmm(tensor1, tensor2)`：批的矩阵相乘，即b\*m\*n和b\*n\*p的张量相乘，得到b\*m\*p结果
+ `torch.mul()`或`tensor1 * tensor2`：按位点乘
+ `torch.cat(list_of_tensors, dim=0)`：拼接张量，dim=0为上下拼接，dim=1为左右拼接

### 索引

+ `y=x[0, :];y += 1`会使得x也一起变化，==共享内存==

#### 自动求导

+ `tensor1.requires_grad_(True)`：设置为True后，张量的requires_grad属性为True，才能追踪张量的计算，之后使用链式法则求导以传播梯度。模型中定义的张量默认为True，用户定义的张量默认为False
+ `tensor1.gard`：输出tensor1的梯度
+ `loss.backward() `：计算完loss后完成所有梯度计算
+ `with torch.no_grad()`：评估模型时使用，不计算所有梯度，加速并减少内存

### 模型

#### 定义

通常继承`nn.Module`来写自己的layer或model，并重载\_\_init\_\_和forward函数，基本结构为：

```python
class xxxNet(nn.Module):
    def __init__(self, xxx):
        super(xxxNet, self).__init__()
        self.xxx = nn.xxx(xxx)
    
    def forward(self, x):
        y = self.xxx(x)
        return y
# 或
xxxNet = nn.Sequential(
		nn.xxx(),
		nn.xxx()
		)
```

#### 参数

`xxxNet.named_parameters()`可查看所有可学习的参数，打印时需要for循环单独打印每个参数的name和param

```python
for name, param in model.named_parameters():
    print(name, param.size())
```

此外，层中的`nn.Prarmeter`会被自动加入到参数列表中，但`torch.rand`等其他tensor不会加入参数列表

在模型训练结束后，可以打印模型或优化器的参数

```python
print(model.state_dict())
print(optimizer.state_dict())
```

#### 状态

+ `model.train()`：训练状态
+ `model.eval()`：测试/验证状态，==bn固定参数，dropout失效==，使用训练好的值

#### GPU

`model.cuda()`将模型放到GPU，模型中定义的参数也会放在GPU，但模型传入的参数仍未cpu。注意不需要`model=model.cuda()`

#### 保存和加载模型

保存模型

```python
torch.save(model.state_dict(), PATH) # 推荐的文件后缀名是pt或pth
# 或
torch.save(model, PATH)
```

加载模型

```python
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
# 或
model = torch.load(PATH)
```

推荐前者

### 代码搭建

```python
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


# 读取数据
test_x = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")
train_y = train['label']
train_x = train.drop('label', axis=1)

# 查看数据前几行预览
print(train_x.head())
# 查看数据集详情
print(train_x.describe())
# 查看某一列的数据集分布
print(train_x['pixel0'].value_counts())
# 查看某一列的数据集分布图
sns.countplot(train_y)
plt.show()
# 对特定某行转化为1和0
train_x['Sex'] = pd.factorize(train_x.Sex)[0]
# 查看nan
print("train_x describe: \n", train_x.isnull().any().describe())
print("\ntest_x describe: \n", test_x.isnull().any().describe())
print("\ntrain_y describe: \n", train_y.value_counts())
# 将缺失数据替换为平均值
if np.isnan(train_x.astype(float)).sum() > 0:
    print("NaN exists in train_X.")
    train_x = train_x.fillna(train_x.mean())

# 归一化
train_x /= 255
test_x /= 255

# 改变维度
train_x = train_x.values.reshape(-1, 1, 28, 28)
test_x = test_x.values.reshape(-1, 1, 28, 28)

# 可视化
def set_figsize(figsize=(3.5, 2.5)):
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = figsize
set_figsize()
plt.scatter(train_x[:, 0].numpy(), train_y.numpy(), 1)
plt.show()

# 划分训练集和验证集。为防止数据不均衡，设置stratify可以保证训练集和验证集里各类比例相同
train_x, validation_x, train_y, validation_y = train_test_split(train_x, train_y, test_size=0.1, stratify=train_y)

# 对y从0-9转化为独热码，结果为np.array。具体是否需要独热吗，根据loss函数确定，二分类BCELoss需要，多分类CELoss不需要。
train_y = to_categorical(train_y, num_classes=10)
validation_y = to_categorical(validation_y, num_classes=10)

# 定义参数
num_epochs = 10
num_classes = 10
learning_rate = 0.01
batch_size = 100

# gpu
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)
if not args.cuda:
    exit()

# 数据变为tensor
train_x = torch.FloatTensor(np.array(train_x))
validation_x = torch.FloatTensor(np.array(validation_x))
test_x = torch.FloatTensor(np.array(test_x))
train_y = torch.LongTensor(np.array(train_y))
validation_y = torch.LongTensor(np.array(validation_y))

# 分batch
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
validation_dataset = torch.utils.data.TensorDataset(validation_x, validation_y)
test_dataset = torch.utils.data.TensorDataset(test_x)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型（CNN）
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        # 28*28*1
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=3),  # 28*28*8
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14*14*8
            nn.Dropout(0.5))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=7, stride=1, padding=3),  # 14*14*16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 7*7*16
            nn.Dropout(0.5))

        self.fc = nn.Linear(7*7*16, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # 即将batch*7*7*32的数据集改为batch*(7*7*32)的大小，进入全连接层
        out = self.fc(out)
        return out

# 实例化模型
model = ConvNet(num_classes)

# tensor放在GPU
train_x = train_x.cuda()
validation_x = validation_x.cuda()
test_x = test_x.cuda()
train_y = train_y.cuda()
validation_y = validation_y.cuda()
model.cuda()

# 定义其他参数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []
total_train_step = len(train_loader)
total_validation_step = len(validation_loader)

# 训练和验证
for epoch in range(num_epochs):
    correct = 0
    for step, (train_x, train_y) in enumerate(train_loader):
        train_x, train_y = train_x.cuda(), train_y.cuda()  # 重要
        model.train()
        outputs = model(train_x)
        train_loss = criterion(outputs, train_y)
        correct += (torch.max(outputs, 1)[1] == train_y).sum().item()

        optimizer.zero_grad()  # 每个batch清零梯度，因为batch之间梯度不需要累积
        train_loss.backward()  # 计算完loss后完成所有梯度计算
        optimizer.step()  # 更新参数
        
        if (step + 1) % 100 == 0:
            print("step[{}/{}], loss:{:.4f}".format(step + 1, total_train_step, train_loss))

    train_losses.append(train_loss.item())
    train_acc = correct / (total_train_step * batch_size)
            
    model.eval()  # 测试模式，不更新参数
    correct = 0
    for step, (validation_x, validation_y) in enumerate(validation_loader):
        if args.cuda:
        	validation_x, validation_y = validation_x.cuda(), validation_y.cuda()
        outputs = model(validation_x)
        validation_loss = criterion(outputs, validation_y)
        correct += (torch.max(outputs, 1)[1] == validation_y).sum().item()
    validation_acc = correct / (total_validation_step * batch_size)
    print('epoch[{}/{}], train loss:{:.4f}, train acc:{:.4f}, validation loss:{:.4f}, validation acc:{:.4f}'.format(epoch + 1, num_epochs, train_loss.item(), train_acc, validation_loss.item(), validation_acc))

# 测试
model.eval()
with torch.no_grad():  # 关闭自动求导机制，加速计算，减小存储
    for step, test_x in enumerate(test_loader):
        if args.cuda:
        	test_x = test_x[0].cuda()
        output = model(test_x)
        if not step:
            test_y = outputs
        else:
            test_y = torch.cat((test_y, output), 0)
# 写入数据到文件
if torch.cuda.is_available():
    test_y = test_y.cpu()
test_y = pd.DataFrame(torch.argmax(test_y, 1).numpy())
test_id = pd.DataFrame([i for i in range(1, test_y.shape[0] + 1)])
test_y = pd.concat([test_id, test_y], axis=1)
test_y.columns = ['ImageId', 'Label']
test_y.to_csv('submission.csv', index=False, encoding='utf8')

# loss可视化
plt.plot([i + 1 for i in range(num_epochs)], train_losses, 'r-', lw=1)
plt.yticks([x * 0.1 for x in range(15)])
plt.show()
```

### 多卡

```python
model = xxxx

# 增加这一段
if torch.cuda.device_count() > 1:
    print("Using {} gpu".format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

model.cuda()
```

运行时`#SBATCH --gres=gpu:V100:2`即改为2个gpu，`CUDA_VISIBLE_DEVICES=0,1 python train.py`即在0号基础上增加1号gpu

+ 注意，forward的参数都会认为和batchsize有关，因此会被拆分给各卡，默认按第一个维度拆分。如果不想被拆分，请在模型定义时传入，模型中用self.xxx形式使用。

+ model.load_state_dict可能会load到一张卡上，需要解决

### 其他细节

+ 神经网络的梯度下降法如何更新参数

    + 计算batch的平均损失，每个需要更新的参数$var=var-\frac{\alpha}{batch\_size}*\frac{\partial loss}{\partial var}$

+ 为什么全连接层后往往需要激活函数

    + 如果没有激活函数，多个全连接层连接，相当于多个线性变换，合并后就是一个线性变换，毫无意义

+ 训练误差和泛化误差

    + 泛化误差即测试集上的误差
    + 显然，训练误差一般小于或等于泛化误差，即模型不会在验证集上表现的比训练集还好
    + 深度学习显然需要降低泛化误差，但是训练只能减小训练误差，因此需要关注过拟合问题

+ model在train之后，需要验证cv集，为什么反传的loss是train的而不是cv的？

    + cv集的作用就是查看表现，由于test没有label，只能通过找一个不会用来训练、模型从没见过的数据集（即cv集）来进行验证效果，但模型还是应该根据train集来训练。

+ 归一化

    + **先划分后归一化，但只是用训练集进行归一化的fit**，保证不从测试集得到任何数据。在有交叉验证集的数据，也应该**只根据训练集**归一化，在交叉验证集调到最佳参数，在测试集上测试。且每一个特征独自进行归一化，保证各个特征最后都在相同量级内。
    + 具体方法
        + 均值-方差归一化：$x=\frac{x-\mu}{\sigma}$，结果符合标准正态分布，均值为0方差为1，最大最小值没有范围，常用于距离相关的算法如K-means。
        + 最大-最小归一化：$x=\frac{x-x_{min}}{x_{max}-x_{min}}$，结果都在$[0,1]$范围内，常用于图像处理（都在0-255之内）。由于测试数据可能在训练集的最值之外，使用最大-最小归一化可能导致测试数据归一化后不在$[0,1]$范围内，出现问题。

+ 混淆矩阵：二分类问题时可用

    ![混淆矩阵](http://ww1.sinaimg.cn/large/96803f81ly1fzf7rkjiqaj20d406oglx.jpg)

    ```python
    # confusion_matrix混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(cv_Y, predict_cv_Y)
    print(cm)
    ```

+ 准确率：

    ![准确率](http://ww1.sinaimg.cn/large/96803f81ly1fzf7ucwtzaj207w02rt8m.jpg)

    ```python
    # 准确率
    from sklearn.metrics import accuracy_score
    score = accuracy_score(cv_Y, predict_cv_Y)
    ```

+ F1值：

    ![F1值](http://ww1.sinaimg.cn/large/96803f81ly1fzf7y11jopj2095034t8q.jpg)

    ```python
    # f1_score， 用于二分类
    from sklearn.metrics import f1_score
    score = f1_score(cv_Y, predict_cv_Y)
    print(cm)
    ```

+ 均方误差MSE：

    ![均方误差](http://ww1.sinaimg.cn/large/96803f81ly1fzgd80u212j207a034wed.jpg)

    ```python
    # 均方误差MSE
    from sklearn.metrics import mean_squared_error
    score = mean_squared_error(cv_Y, predict_cv_Y)
    print(score)
    ```

+ 决定系数r2：回归问题常用

    ![决定系数](http://ww1.sinaimg.cn/large/96803f81ly1fzgd9y3y8vj206i01ja9y.jpg)

    ```python
    # 决定系数，可用于回归
    from sklearn.metrics import r2_score
    score = r2_score(cv_Y, predict_cv_Y)
    ```

+ 过拟合解决方法

    + dropout：有一定概率丢弃每个数值，==测试时一般不使用Dropout==，调用model.eval()即可
    + L2正则化或权重衰减：利用正则化系数$\lambda$作为惩罚，以放大逻辑回归的梯度中每个$\theta$的影响，从而减小$\theta$的值，防止过拟合。$\theta_0$恒为1，不需惩罚。==L1正则化是绝对值之和，L2正则化是平方和的开方==，常用L2来防止过拟合。
    + batch normalizatin：`nn.BatchNorm1d(x)`，经过每一层计算，数据分布会发生变化，学习越来越困难。BN即在一个mini-batch内对数据的每个feature进行均值-方差归一化，归一化后再做一个线性变换$y=\gamma x+\beta$，通过两个可学习的参数进行拉伸和偏移，一定程度上保留数据原本特征。极端的情况下，$\gamma$和$\beta$就是方差和均值，数据完全还原。batch_size一般塞满卡即可。==BN层在层和激活函数之间==。在测试时，为了在batch的均值方差不一样时，结果能一样，通常取整个数据集的均值方差，并==固定==
    + 减小调整模型复杂度：模型越复杂，越容易过拟合。如高阶多项式比低阶多项式更容易拟合到训练集
    + 增大数据量：数据量过少时，容易过拟合。较难实现
    + 提前停止Early Stop

+ 梯度爆炸和消失

    + BP反传中，多个导数连乘可能导致梯度非常小，无法更新参数，导致梯度消失。同理，可能梯度非常大，导致梯度爆炸。将Sigmoid换用ReLU或LeakyReLU激活函数可解决，其正数的导数恒为1，不会带来梯度爆炸和消失。

+ loss

    + `nn.CrossEntropyLoss(outputs, labels)`：n分类问题pred为m\*n维，y为m\*==1==维（`LongTensor`）。过程为先进行`nn.LogSoftmax()`，再`nn.NLLLoss()`，其中`nn.NLLLoss()`即为对于每个样本，`loss += -log(pred)[class]`，求$loss=-\Sigma_{i=0}^n y_i\log pred_i$，y为0或1，总loss为$\frac{1}{m}\Sigma loss$。==交叉熵只要满足label为1的概率足够大，就能保证分类正确，因此不考虑label为0的pred==
    + `nn.BCEWithLogitsLoss(outputs, labels)`：二分类问题pred为m\*n维，y为m\*==n==维独热码（`FloatTensor`）。过程为先进行`nn.Sigmoid()`，再`nn.BCELoss()`，其中`nn.BCELoss()`即为对于每个样本，求$loss=-\frac{1}{n}\Sigma_{i=0}^n(y_i\ln pred_i+(1-y_i)\ln(1-pred_i))$，总loss为$\frac{1}{m}\Sigma loss$。
    + `nn.MSELoss()`：均方误差

+ 激活函数：加入非线性运算，否则所有全连接层合并后等价于一个线性变换

    + ReLU速度快。
    + ReLU效果不好时，尝试LeakyReLU或Maxout。
    + 模型不深，激活函数作用不大。

+ 目标函数J

    + 目标函数J时优化时的概念，一般等同于损失函数loss或loss+L2正则化惩罚项
    
+ 优化器

    + Adam：为每一个参数适应性地保留1个学习率，不像SGD一样学习速率不变

## ANN

+ 人工神经网络ANN是最简单的深度学习算法之一。模型包含一个输入层，若干个隐藏层和一个输出层。隐藏层越多效果越好，训练难度也越大，全连接神经网络隐藏层一般不超过三层。每层有若干个神经元，隐藏层神经元个数可以用$\sqrt{n*l}$来作为初始值，其中n和l为输入层和输出层的神经元个数。

    ![ANN](http://ww1.sinaimg.cn/large/96803f81ly1g06zla4re8j20if09cq73.jpg)

    神经元之间采用全连接，连接上有权重w。当要计算下一层某个神经元的值时，所用公式为$y_1=sigmoid(w^Ta)$。即每个要计算的神经元的值等于上一层与它连接的所有神经元与连接上的权重的乘积和，输入到sigmoid的结果，只需要计算出权重即可。

+ 权重由反向传播更新。

    + 对于输出层，求得误差$\delta_i=y_{i,true}-y_i$。对于隐藏层，求得误差$\delta_i=\Sigma w\delta$。
    + 更新权重。从输入层开始，$w=w+\eta\delta\frac{\partial y}{\partial\Sigma w} x$，其中$\eta$是学习速率。

+ ```python
    num_epochs = 500
    learning_rate = 0.02
    
    class AnnNet(nn.Module):
        def __init__(self):
            super(AnnNet, self).__init__()
            self.fc1 = nn.Linear(7, 5)
            self.tanh = nn.Tanh()
            self.fc2 = nn.Linear(5, 5)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(5, 2)
    
        def forward(self, x):
            out = self.fc1(x)
            out = self.tanh(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            return out
    
    
    model = AnnNet()
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        losses.append(loss)
        loss.backward()
        optimizer.step()
        correct = (torch.max(pred.data, 1)[1] == train_y).sum()
        acc = correct.item() / len(pred)
        print("Epoch:[{}/{}], acc: {:.4f}, Loss:{:.4f}".format(epoch + 1, num_epochs, acc, loss.item()))
    
    model.eval()
    with torch.no_grad():
        test_y = model(test_x)
    test_y = pd.DataFrame(torch.argmax(test_y, 1).numpy())
    test_y = pd.concat([test_id, test_y], axis=1)
    test_y.columns = ['PassengerId', 'Survived']
    test_y.to_csv('submission.csv', index=False, encoding='utf8')
    ```

## CNN

+ 一个CNN往往由**N个卷积层和ReLU层叠加，再加一个可选的池化层，将该结构重复M次，最后叠加K个全连接层**，`INPUT -> [[CONV -> ReLU]*N -> POOL?]*M -> [FC]*K`。

+ 卷积层（Convolutional Layer）：卷积层通过滤波器（filter）将图片卷积为Feature Map，即从原始图像提取出的特征，这个过程成为卷积，是一种局部连接的网络。Feature Map计算公式如下：

    ![Feature Map](https://i.loli.net/2019/06/18/5d08e73aecf1920876.jpg)

    其中w为滤波器中第m行n列的权重，w为偏置，x为图片的每个像素，D为图片深度，F为Filter的边长。对于一个5\*5的图片，设置滤波器大小为3\*3，步幅stride为1，偏置为0，则得到的Feature Map大小为3\*3，计算过程如下。

    ![滤波器](https://i.loli.net/2019/06/18/5d08e57a404ea88450.gif)

    Feature Map大小W_2、Filter大小W_1、步幅S、零填充P满足如下公式：

    ![Feature Map大小](https://i.loli.net/2019/06/18/5d08e6bd6f02080619.jpg)

    其中零填充是在图片外侧填充P层0，有利于图像边缘的特征提取。

+ 池化层（Pooling Layer）:池化pooling即汇总，作用是下采样，去掉Feature Map中不必要的参数。常用Max Pooling，Mean Pooling等。

    + Max Pooling是对每个部分取最大值。一个2\*2的Max Pooling如图：

        ![Max Pooling](https://i.loli.net/2019/06/19/5d091615af29320884.jpg)

    对于深度为D的Feature Map，各层独立做Pooling，因此Pooling后的深度仍然为D。池花层的采样思想，缓解了对同一类物体中像素位置的要求程度

+ 全连接层：计算梯度，反向传播更新权重。

    ```python
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms
    
    x_train = torchvision.datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    x_test = torchvision.datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=x_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=x_test, batch_size=batch_size, shuffle=False)
    
    
    class ConvNet(nn.Module):
        def __init__(self, num_classes):
            super(ConvNet, self).__init__()
            # 28*28*1
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3),  # 28*28*16
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 14*14*16
                nn.Dropout(0.5))
    
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),  # 14*14*32
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 7*7*32
                nn.Dropout(0.5))
    
            self.fc = nn.Linear(7*7*32, num_classes)
    
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out
    ```

+ CNN为什么可用于CV，NLP，speech等领域：以上领域都存在局部和整体的关系，低层次特征组合为高层次特征的共性。CNN通过卷积、池化等实现此共性。

## 集成学习

+ 以Boosting方法为例。所谓Boosting方法，即迭代提升，故为串行算法。先训练弱学习器，然后根据前一个弱学习器分错的样本，改变样本的概率分布构成新的训练集，从而训练出一个更强的学习器。这样反复迭代提升，就能得到一系列分类器。最后，将这些分类器组合起来，就能构成一个很强的学习器。

### AdaBoost

+ 在不改变训练数据的情况下，通过在迭代训练弱学习器中，不断提升被错分类样本的权重（也就是使被错分的样本在下一轮训练时得到更多的重视），不断减少正确分类样本的权重。最后通过加权线性组合M个弱分类器得到最终的分类器，正确率越高的弱分类器的投票权数越高，正确率低的弱分类器自然投票权数就低。

### GBDT

+ 迭代的决策树算法，由多棵决策树组成，所有树的结论累加起来做最终答案。GBDT中的树是回归树（不是分类树），GBDT用来做回归预测。

### xgboost

+ 相比GBDT，使用了正则化防止过拟合，loss采用了泰勒展开实现二阶导数而不是一阶导数，寻找最佳分割点标准是最大化Lsplit。xgboost在精度和效率上都有了提升。