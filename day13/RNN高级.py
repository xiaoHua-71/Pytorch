"""LSTM 时间序列回归示例：用连续 30 个点预测下一个点。"""

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


WINDOW_SIZE = 30
TRAIN_RATIO = 0.8


def make_series(length=700):
    """生成带噪声的周期序列，模拟传感器或销量数据。"""
    generator = torch.Generator().manual_seed(21)
    time = torch.arange(length, dtype=torch.float32)
    clean = torch.sin(time * 0.08) + 0.3 * torch.sin(time * 0.19)
    noise = 0.08 * torch.randn(length, generator=generator)
    return clean + noise


def make_windows(series, window_size):
    inputs = []
    targets = []
    for start in range(len(series) - window_size):
        inputs.append(series[start : start + window_size].unsqueeze(-1))
        targets.append(series[start + window_size])
    return torch.stack(inputs), torch.stack(targets).unsqueeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.15,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return self.head(output[:, -1, :])


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs)
            total_loss += criterion(predictions, targets).item() * len(targets)
            total_count += len(targets)
    return total_loss / total_count


def main():
    torch.manual_seed(21)
    series = make_series()

    # 先按时间切分，再只用训练部分计算标准化参数，防止未来信息泄漏。
    split_point = int(len(series) * TRAIN_RATIO)
    train_raw = series[:split_point]
    mean = train_raw.mean()
    std = train_raw.std().clamp_min(1e-6)
    normalized = (series - mean) / std

    inputs, targets = make_windows(normalized, WINDOW_SIZE)
    train_count = split_point - WINDOW_SIZE
    train_inputs, test_inputs = inputs[:train_count], inputs[train_count:]
    train_targets, test_targets = targets[:train_count], targets[train_count:]
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets), batch_size=32, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_inputs, test_targets), batch_size=64, shuffle=False
    )

    print("input shape (batch, time, feature):", tuple(train_inputs.shape))
    model = LSTMRegressor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_targets)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            train_loss = total_loss / len(train_inputs)
            test_loss = evaluate(model, test_loader, criterion)
            print(
                f"epoch={epoch + 1:02d}, train_mse={train_loss:.4f}, "
                f"test_mse={test_loss:.4f}"
            )

    model.eval()
    with torch.no_grad():
        normalized_prediction = model(test_inputs[:5]).squeeze(-1)
    prediction = normalized_prediction * std + mean
    actual = test_targets[:5].squeeze(-1) * std + mean
    print("first five predictions:", [round(value, 3) for value in prediction.tolist()])
    print("first five actual values:", [round(value, 3) for value in actual.tolist()])
    print("last known value:", round(float(series[-1]), 3))
    print("next-step demo input period:", round(2 * math.pi / 0.08, 2), "steps")


if __name__ == "__main__":
    main()
