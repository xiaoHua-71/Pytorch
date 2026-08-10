import numpy as np


# Training data: the target relationship is y = 2x.
x_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
y_data = np.array([2.0, 4.0, 6.0], dtype=np.float64)


def forward(x, w, b):
    """线性模型"""
    return w * x + b


def mean_squared_error(x, y, w, b):
    """均方误差 MSE"""
    return np.mean((forward(x, w, b) - y) ** 2)


def gradient_descent_step(x, y, w, b, learning_rate):
    """梯度下降算法"""
    y_pred = forward(x, w, b)
    error = y_pred - y

    # Gradients of MSE with respect to the weight and bias.
    grad_w = np.mean(2 * error * x)
    grad_b = np.mean(2 * error)

    # Move in the direction opposite to the gradient to reduce the loss.
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    return w, b, grad_w, grad_b


learning_rate = 0.01
epochs = 10000
w = 0.0
b = 0.0

for epoch in range(epochs):
    w, b, grad_w, grad_b = gradient_descent_step(
        x_data, y_data, w, b, learning_rate
    )

    if epoch < 5 or (epoch + 1) % 100 == 0:
        current_loss = mean_squared_error(x_data, y_data, w, b)
        print(
            f"epoch={epoch + 1:4d}, loss={current_loss:.10f}, "
            f"grad_w={grad_w:.10f}, grad_b={grad_b:.10f}, "
            f"w={w:.10f}, b={b:.10f}"
        )


print("\nTraining complete")
print(f"Final model: y = {w:.10f}x + {b:.10f}")
print("Predictions:", np.round(forward(x_data, w, b), 10))
print("Targets:    ", y_data)
