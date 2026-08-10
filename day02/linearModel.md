# 线性模型 Linear Model

## 机器学习的四步流程

1、选择数据集 DataSet

训练数据是一组 (x, y) 样本对。本例用一个最简单的数据集：

| x | y |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |

它背后隐藏的目标关系是 `y = 2x`，但我们不让模型直接看到这个关系——模型只能从数据里自己学出来。

2、选择模型

模型就是"用什么样的函数形式把 x 映射到 y"。先选最简单的一元线性模型：

```text
y_pred = w * x
```

其中 `w` 是模型的参数（权重）。模型能学习的东西，就是这个 `w`。

> 更一般的形式是带偏置的 `y_pred = w * x + b`，只多一个参数 b，思想完全一样，只是搜索空间多一维。为了直观，本节先用只有一个参数 w 的版本。

3、Training

训练就是：用数据集确定参数 w，使得模型的预测 y_pred 尽量接近真实 y。怎么衡量"接近不接近"？——用损失函数，见下文。

4、inferring 应用、推理

训练完成后，模型就可以对没见过的 x 做预测（推理）。比如学出 w ≈ 2 后，输入 x = 4，模型能预测出 y ≈ 8。

## 损失函数：均方误差（MSE）

损失函数（Loss Function）衡量当前模型有多差。对线性模型，最常用的是**均方误差（Mean Squared Error）**。

单个样本的误差用平方差表示：

```text
loss_i = (y_pred - y)²
```

为什么用平方而不是绝对值？

- 平方永远非负，正误差、负误差都能被计入；
- 平方会**放大大的偏差**，让模型优先处理错得离谱的样本。

对一批样本取平均，得到 MSE：

```text
MSE = mean((y_pred - y)²)
```

以 `w = 1` 为例，手工算一遍：

| x | y | y_pred = 1·x | (y_pred − y)² |
|---|---|---|---|
| 1 | 2 | 1 | 1 |
| 2 | 4 | 2 | 4 |
| 3 | 6 | 3 | 9 |

```text
MSE = (1 + 4 + 9) / 3 ≈ 4.667
```

MSE 越小，说明这条直线越贴合数据。

## 常用的求"最优直线"的方法：穷举法

有了损失函数，问题就变成：**找一个 w，让 MSE 最小。**

最朴素的方法是**穷举法**：在 w 的可能取值范围内，按一定步长逐个尝试，算出每个 w 的 MSE，取 MSE 最小的那个。

```python
import numpy as np
import matplotlib.pyplot as plt

# 训练数据：目标关系是 y = 2x
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x, w):
    """线性模型：y_pred = w * x"""
    return x * w


def loss(x, y, w):
    """单个样本的平方误差"""
    y_pred = forward(x, w)
    return (y_pred - y) ** 2


w_list = []    # 尝试过的 w
mse_list = []  # 每个 w 对应的 MSE

# 穷举：在 w ∈ [0, 4] 上以步长 0.1 逐个尝试
for w in np.arange(0.0, 4.1, 0.1):
    l_sum = 0.0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val, w)
        loss_val = loss(x_val, y_val, w)
        l_sum += loss_val
        print(f"x={x_val} y={y_val} y_pred={y_pred_val:.2f} loss={loss_val:.4f}")
    mse = l_sum / len(x_data)
    print(f"w={w:.2f} MSE={mse:.4f}\n")
    w_list.append(w)
    mse_list.append(mse)
```

穷举法的优点：

- 简单直观，不需要求导数；
- 对单个参数（一维搜索）完全够用。

## 运行结果

当尝试到 `w = 2.0` 时，预测值与真实值完全一致，MSE 为 0：

```text
x=1.0 y=2.0 y_pred=2.00 loss=0.0000
x=2.0 y=4.0 y_pred=4.00 loss=0.0000
x=3.0 y=6.0 y_pred=6.00 loss=0.0000
w=2.00 MSE=0.0000
```

穷举的最小 MSE 出现在 w ≈ 2，模型学到了目标关系 `y = 2x`。

## 可视化：loss(w) 曲线

把每个 w 对应的 MSE 画出来，横轴是 w、纵轴是 MSE：

```python
plt.plot(w_list, mse_list)
plt.xlabel("w")
plt.ylabel("MSE")
plt.title("Loss vs. w")
plt.show()
```

曲线呈一个**碗形（凸函数）**，谷底在 w ≈ 2 处，对应损失最小。直观理解：

- w 太小（如 1），直线 `y = x` 斜率不够，预测普遍偏低，MSE 大；
- w 太大（如 3），直线 `y = 3x` 偏陡，预测普遍偏高，MSE 也大；
- 只有 w ≈ 2 时，直线 `y = 2x` 正好穿过全部数据点。

## 小结与局限

- 机器学习四步：**选数据集 → 选模型 → 训练（最小化损失）→ 推理**。
- 线性模型 `y = w·x` 用 MSE 衡量好坏，穷举法能在一维参数空间找到最优 w。

但穷举法有根本局限：

- 是**离散**搜索，结果取决于步长，精度有限；
- 参数一多（w、b……），搜索空间**指数级增长**，穷举根本不现实。

所以下一步需要一种能**连续地、自动地**更新参数的方法——这就是 day03 要讲的**梯度下降（Gradient Descent）**。本目录的 `LossFunction.py` 已经给出了一种基于梯度下降的训练实现。
