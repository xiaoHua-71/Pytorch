import numpy as np


# Training data: the target relationship is y = 2x.
x_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
y_data = np.array([2.0, 4.0, 6.0], dtype=np.float64)


def forward(x, w, b):
    """前向传播：z = wx + b（线性模型）"""
    return w * x + b


def loss(z, y):
    """前向传播的最后一环：MSE 损失 L = (z - y)²"""
    return np.mean((z - y) ** 2)


def backward(x, y, z):
    """反向传播：用链式法则从 L 往参数方向逐层求梯度。

    计算图：L <- z <- (w, b)
    - dL/dz = 2(z - y)            上游梯度
    - dz/dw = x,  dz/db = 1       局部梯度
    - dL/dw = dL/dz * dz/dw = 2(z-y) * x
    - dL/db = dL/dz * dz/db = 2(z-y)
    """
    dL_dz = 2 * (z - y)                 # 上游梯度
    dz_dw = x                           # 局部梯度
    dz_db = np.ones_like(x)

    dL_dw = np.mean(dL_dz * dz_dw)      # = mean(2(z-y)*x)
    dL_db = np.mean(dL_dz * dz_db)      # = mean(2(z-y))
    return dL_dw, dL_db


def gradient_descent_step(w, b, dL_dw, dL_db, learning_rate):
    """梯度下降：拿反向传播求出的梯度更新参数。"""
    w -= learning_rate * dL_dw
    b -= learning_rate * dL_db
    return w, b


learning_rate = 0.01
epochs = 10000
w = 0.0
b = 0.0

for epoch in range(epochs):
    # 1. 前向传播：算出预测值 z，并记录中间变量
    z = forward(x_data, w, b)

    # 2. 反向传播：从损失出发，用链式法则逐层求出 dL/dw、dL/db
    dL_dw, dL_db = backward(x_data, y_data, z)

    # 3. 梯度下降：用梯度更新参数
    w, b = gradient_descent_step(w, b, dL_dw, dL_db, learning_rate)

    if epoch < 5 or (epoch + 1) % 100 == 0:
        current_loss = loss(z, y_data)
        print(
            f"epoch={epoch + 1:4d}, loss={current_loss:.10f}, "
            f"grad_w={dL_dw:.10f}, grad_b={dL_db:.10f}, "
            f"w={w:.10f}, b={b:.10f}"
        )


print("\nTraining complete")
print(f"Final model: y = {w:.10f}x + {b:.10f}")
print("Predictions:", np.round(forward(x_data, w, b), 10))
print("Targets:    ", y_data)
