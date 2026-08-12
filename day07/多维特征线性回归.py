import torch


torch.manual_seed(0)


def prepare_data():
    """准备多维特征数据，X 的形状为 (样本数, 特征数)。"""
    x_data = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [4.0, 2.0, 2.0],
            [3.0, 0.0, 4.0],
            [1.0, 4.0, 2.0],
            [5.0, 1.0, 3.0],
            [2.0, 3.0, 5.0],
        ],
        dtype=torch.float32,
    )

    # 目标关系：y = 2*x1 - 3*x2 + 0.5*x3 + 4
    # 使用 0:1、1:2、2:3 切片，使 y_data 保持 (N, 1) 形状。
    y_data = (
        2 * x_data[:, 0:1]
        - 3 * x_data[:, 1:2]
        + 0.5 * x_data[:, 2:3]
        + 4
    )
    return x_data, y_data


class MultiFeatureLinearModel(torch.nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.linear = torch.nn.Linear(feature_count, 1)

    def forward(self, x):
        return self.linear(x)


def train(model, x_data, y_data, epochs=5000):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(epochs):
        prediction = model(x_data)
        loss = criterion(prediction, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"epoch={epoch:4d}, loss={loss.item():.8f}")


def predict(model, features):
    """预测一个或多个样本，features 的形状应为 (N, 3)。"""
    model.eval()
    with torch.no_grad():
        return model(features)


def main():
    x_data, y_data = prepare_data()
    print("x_data.shape =", x_data.shape)
    print("y_data.shape =", y_data.shape)

    feature_count = x_data.shape[1]
    model = MultiFeatureLinearModel(feature_count)
    print("\nmodel:")
    print(model)

    print("\n开始训练：")
    train(model, x_data, y_data)

    learned_weight = model.linear.weight.detach().squeeze()
    learned_bias = model.linear.bias.detach().item()
    print("\n学习结果：")
    print("weight =", learned_weight)
    print("bias   =", learned_bias)
    print("目标参数：weight = tensor([2.0000, -3.0000, 0.5000]), bias = 4.0")

    # 单样本也要保留 batch 维度，形状为 (1, 3)。
    one_sample = torch.tensor([[1.0, 2.0, 3.0]])
    one_prediction = predict(model, one_sample)
    expected_value = 2 * 1.0 - 3 * 2.0 + 0.5 * 3.0 + 4
    print("\n单样本预测：")
    print("input      =", one_sample)
    print("prediction =", one_prediction.item())
    print("expected   =", expected_value)

    # 批量预测：2 个样本、每个样本 3 个特征，输入形状为 (2, 3)。
    batch_samples = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [3.0, 1.0, 4.0],
        ]
    )
    batch_predictions = predict(model, batch_samples)
    print("\n批量预测：")
    for features, prediction in zip(batch_samples, batch_predictions):
        print(f"features={features.tolist()}, prediction={prediction.item():.4f}")


if __name__ == "__main__":
    main()
