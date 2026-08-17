import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(0)


def make_dataset(samples_per_class=100):
    centers = torch.tensor(
        [
            [-2.0, -1.5],
            [2.0, -1.0],
            [0.0, 2.0],
        ]
    )

    features = []
    labels = []
    for class_index, center in enumerate(centers):
        class_features = center + 0.6 * torch.randn(samples_per_class, 2)
        class_labels = torch.full(
            (samples_per_class,), class_index, dtype=torch.long
        )
        features.append(class_features)
        labels.append(class_labels)

    return torch.cat(features), torch.cat(labels)


class MulticlassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        return self.network(x)


def main():
    x_data, y_data = make_dataset()
    dataset = TensorDataset(x_data, y_data)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MulticlassModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        total_correct = 0

        for inputs, targets in train_loader:
            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.shape[0]
            total_correct += (logits.argmax(dim=1) == targets).sum().item()

        if epoch % 10 == 0 or epoch == 99:
            average_loss = total_loss / len(dataset)
            accuracy = total_correct / len(dataset)
            print(
                f"epoch={epoch:3d}, "
                f"loss={average_loss:.4f}, accuracy={accuracy:.2%}"
            )

    model.eval()
    test_inputs = torch.tensor(
        [
            [-2.0, -1.0],
            [2.0, -1.5],
            [0.0, 2.5],
        ]
    )

    with torch.no_grad():
        test_logits = model(test_inputs)
        test_probabilities = torch.softmax(test_logits, dim=1)
        test_predictions = test_logits.argmax(dim=1)

        train_predictions = model(x_data).argmax(dim=1)
        confusion_matrix = torch.zeros(3, 3, dtype=torch.long)
        for target, prediction in zip(y_data, train_predictions):
            confusion_matrix[target, prediction] += 1

    print("\nprobabilities:")
    print(test_probabilities)
    print("predicted classes:", test_predictions.tolist())
    print("confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix)


if __name__ == "__main__":
    main()
