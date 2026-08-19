"""RNN 最小示例：根据一段序列判断它整体是上升还是下降。"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


SEQUENCE_LENGTH = 20
CLASS_COUNT = 2


def make_dataset(samples_per_class=160):
    """生成两类带噪声的一维序列：上升=0，下降=1。"""
    generator = torch.Generator().manual_seed(7)
    time = torch.linspace(0, 1, SEQUENCE_LENGTH)
    sequences = []
    labels = []

    for label in range(CLASS_COUNT):
        for _ in range(samples_per_class):
            noise = 0.08 * torch.randn(SEQUENCE_LENGTH, generator=generator)
            if label == 0:
                sequence = time + noise
            else:
                sequence = 1 - time + noise
            sequences.append(sequence.unsqueeze(-1))  # (time, feature=1)
            labels.append(label)

    return torch.stack(sequences), torch.tensor(labels, dtype=torch.long)


class SequenceClassifier(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.classifier = nn.Linear(hidden_size, CLASS_COUNT)

    def forward(self, x):
        output, hidden = self.rnn(x)
        last_step = output[:, -1, :]
        return self.classifier(last_step)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            predictions = model(inputs).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / total


def main():
    torch.manual_seed(7)
    sequences, labels = make_dataset()
    print("input shape (batch, time, feature):", tuple(sequences.shape))

    dataset = TensorDataset(sequences, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(8),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    model = SequenceClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(15):
        model.train()
        total_loss = 0.0
        for inputs, batch_labels in train_loader:
            logits = model(inputs)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_labels.shape[0]

        if epoch == 0 or (epoch + 1) % 3 == 0:
            average_loss = total_loss / len(train_set)
            accuracy = evaluate(model, test_loader)
            print(
                f"epoch={epoch + 1:02d}, loss={average_loss:.4f}, "
                f"test_accuracy={accuracy:.2%}"
            )

    model.eval()
    samples = torch.tensor(
        [[[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]] * 3,
         [[1.0], [0.8], [0.6], [0.4], [0.2], [0.0]] * 3],
        dtype=torch.float32,
    )
    with torch.no_grad():
        predictions = model(samples).argmax(dim=1)
    print("sample predictions (0=up, 1=down):", predictions.tolist())


if __name__ == "__main__":
    main()
