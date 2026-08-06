import numpy as np
import matplotlib.pyplot as plt


# Training data: the target relationship is y = 2x.
x_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
y_data = np.array([2.0, 4.0, 6.0], dtype=np.float64)

# Parameters of the linear model: weight w and bias b.
w = 0.0
b = 0.0


def forward(x):
    """Calculate the prediction of the linear model."""
    return w * x + b


def loss(x, y):
    """Calculate the mean squared error between prediction and target."""
    y_pred = forward(x)
    return np.mean((y_pred - y) ** 2)


learning_rate = 0.01
epochs = 10000
losses = []

for epoch in range(epochs):
    # Forward pass and prediction error.
    y_pred = forward(x_data)
    error = y_pred - y_data

    # Gradients of mean squared error with respect to w and b.
    grad_w = np.mean(2 * error * x_data)
    grad_b = np.mean(2 * error)

    # Gradient descent updates the parameters toward lower loss.
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    # Save the loss of this epoch for the training curve.
    current_loss = loss(x_data, y_data)
    losses.append(current_loss)

    if (epoch + 1) % 1000 == 0:
        print(
            f"epoch={epoch + 1:5d}, loss={current_loss:.10f}, "
            f"w={w:.10f}, b={b:.10f}"
        )


print("Training complete")
print(f"Final model: y = {w:.10f}x + {b:.10f}")
print("Predictions:", np.round(forward(x_data), 10))
print("Targets:    ", y_data)

# Plot the loss curve to show whether training converges.
plt.plot(range(1, epochs + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.savefig("day02/training_loss.png")
plt.show()
