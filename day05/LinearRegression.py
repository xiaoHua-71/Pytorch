import torch


# 1. 准备数据：目标关系是 y = 2x（列向量，shape = (3, 1)）
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 2. 设计模型：继承 nn.Module，把网络层放在 __init__，前向计算写在 forward
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # nn.Linear(in_features, out_features)：一个全连接层，内部自动初始化 w、b
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        """前向传播：y_pred = x @ w.T + b，只写 forward，backward 由 autograd 自动完成"""
        return self.linear(x)


model = LinearModel()

# 3. 构造损失函数与优化器
criterion = torch.nn.MSELoss(reduction="sum")  # 课程原写法 size_average=False 已弃用，等价 reduction="sum"
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
for epoch in range(1000):
    y_pred = model(x_data)             # 前向传播：算出预测值
    loss = criterion(y_pred, y_data)   # 计算损失
    if epoch < 5 or (epoch + 1) % 100 == 0:
        print(f"epoch={epoch + 1:4d} loss={loss.item():.8f}")

    optimizer.zero_grad()              # 梯度清零（默认梯度会累加）
    loss.backward()                    # 反向传播：自动求每个参数的梯度
    optimizer.step()                   # 梯度下降：更新参数

print(f"\nw = {model.linear.weight.item():.4f}")
print(f"b = {model.linear.bias.item():.4f}")
print(f"模型: y = {model.linear.weight.item():.4f}x + {model.linear.bias.item():.4f}")
