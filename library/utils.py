import torch
import numpy as np
from torch import nn
from torch.utils import data
from torchvision import transforms, datasets
from IPython import display
from matplotlib import pyplot as plt
from multiprocessing import cpu_count
import time


class Utils:
    """一些公用方法。"""

    @staticmethod
    def evaluate_accuracy_gpu(net, data_iter, device=None):
        """使用 GPU 计算模型在数据集上的精度。"""
        if isinstance(net, torch.nn.Module):
            net.eval()  # 设置为评估模式
            if not device:
                device = next(iter(net.parameters())).device
        # 正确预测的数量，总预测的数量
        metric = Accumulator(2)
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(Utils.accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    @staticmethod
    def train(net, train_iter, test_iter, num_epochs, lr, device):
        """通用模型训练方法"""

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)

        net.apply(init_weights)
        print('training on', device)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss()
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
        timer, num_batches = Timer(), len(train_iter)
        train_l, train_acc, test_acc, metric = None, None, None, None
        for epoch in range(num_epochs):
            # 训练损失之和，训练准确率之和，范例数
            metric = Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], Utils.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(epoch + (i + 1) / num_batches,
                                 (train_l, train_acc, None))
            test_acc = Utils.evaluate_accuracy_gpu(net, test_iter)
            animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
              f'on {str(device)}')

    @staticmethod
    def accuracy(y_hat, y):
        """计算预测正确的数量。"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)
        cmp = y_hat.type(y.dtype) == y
        return float(sum(cmp))

    @staticmethod
    def use_svg_display():
        """使用 svg 格式显示 plot。"""
        display.set_matplotlib_formats('svg')

    @staticmethod
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置 matplotlib 的坐标轴。"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    @staticmethod
    def try_gpu(i=0):
        """尝试获取 GPU，若不存在则返回 CPU。"""
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')


class Animator:
    """在动画中绘制数据。"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        Utils.use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用 lambda 函数捕获参数
        self.config_axes = lambda: Utils.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


class Timer:
    """记录多次运行时间。"""

    def __init__(self):
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和。"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间。"""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """累加器，用于计算 n 个变量的累加和。"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataLoader:
    """数据加载器，用于加载自定义数据集。"""

    def __init__(self, path='../data'):
        self.root_path = path

    def load_data_fashion_mnist(self, batch_size, resize=None):
        """下载 Fashion-MNIST 数据集，然后将其加载到内存中。"""
        trans = [transforms.ToTensor()]
        if resize:
            trans.insert(0, transforms.Resize(resize))
        trans = transforms.Compose(trans)
        mnist_train = datasets.FashionMNIST(
            root=self.root_path, train=True, transform=trans, download=True
        )
        mnist_test = datasets.FashionMNIST(
            root=self.root_path, train=False, transform=trans, download=True
        )
        return (
            data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=cpu_count()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=cpu_count())
        )
