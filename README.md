# DeepEasy: birth for research and fun

![logo](./imgs/logo.png)

「这里的神经元似乎充斥着一股神秘力量。」

## Getting started

### Basic

定义神经网络结构：

```python
nn_architecture: List[Dict] = [
    {   # 第一层
        'input_dim': int,  # 该层每个神经元被连接数
        'output_dim': int, # 该层神经元数
        'activation': str, # 该层激活函数，可选
        'dropout_keep_prob': float  # 该层 dropout 神经元保留概率，可选
    },
    {   # 第二层
        ...
    }
    ...
]
```

实例化对象，指定随机数种子 seed，保证每次随机初始化 weight 的值都相同，便于测试：

```python
from deepeasy.nnet import NeuralNetwork

nn_architecture = [
    {"input_dim": 2, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 1, "activation": "sigmoid"},
]
nn = NeuralNetwork(nn_architecture, seed=100)
```

载入 Mnist 数据集（需要提前下好，解压，放入同一个文件夹）：

```python
from deepeasy.datasets import load_mnist

file_path = '/home/zzzzer/Documents/data/数据集/mnist/'
x_train, y_train, x_test, y_test = load_mnist(file_path)
# x_train.shape=(60000, 28, 28), y_train.shape=(60000,)
# x_test.shape=(10000, 28, 28), y_test.shape=(10000,)
```

查看某一张图片，及其标签：

```python
from PIL import Image

img_idx = 10
# 查看图片
Image.fromarray(x_test[img_idx].reshape(28, 28))
# 查看对应标签
y_test[img_idx]
```

开始训练：
```python
nn.train(
    x_train, y_train, 100,
    batch_size=600,
    lr=0.001,
    gd_name='adam'
)
```

```python
开始训练，迭代次数：100
100
完成！用时：83.32364273071289s
```

画出 Cost、Accuracy 走势：

```python
nn.plot_history()
```

![img](./imgs/01.png)

测试模型：

```python
nn.test_model(x_test, y_test)
```

```python
0.9739
```

### Advance

继续执行 `nn.train()` 方法，在现有模型上继续训练：

```python
nn.train(
    x_train, y_train, 100,
    batch_size=600,
    lr=0.002,
    gd_name='nadam'
)
```

`new_train=True` 清除前面的模型，开启新一轮训练，图像会在原有基础上画出：

```python
nn.train(
    x_train, y_train, 100,
    new_train=True,
    batch_size=600,
    lr=0.002,
    gd_name='sgd'
)
nn.plot_history()
```

![img](./imgs/02.png)

`nn.reset_params()` 清空所有训练记录，回到初始状态，但保留神经网络结构。

## Installation

```python
python3 setup.py install
```

## Include

- Xavier Initializer
- mini batch
- Forward propagation
- Back propagation
- SGD
- Momentum
- RMSprop
- Adam
- Nadam
- inverted dropout
- Cross Entropy Cost
- Mean Squared Cost

## Todo list

- Batch Normalization
- L2 Regularization
- Tests

## References

吴恩达. [深度学习工程师](https://mooc.study.163.com/smartSpec/detail/1001319001.htm). 网易云.

SkalskiP. [ILearnDeepLearning.py](https://github.com/SkalskiP/ILearnDeepLearning.py). GitHub.

斋藤康毅.《深度学习入门：基于Python的理论与实现》. 人民邮电出版社.

