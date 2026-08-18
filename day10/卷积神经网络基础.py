import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


IMAGE_SIZE = 16
NUM_CLASSES = 3


def make_dataset(samples_per_class=200):
    """生成横线、竖线、方块三类简单灰度图片。"""
    generator = torch.Generator().manual_seed(0)
    images = []
    labels = []

    for class_index in range(NUM_CLASSES):
        for _ in range(samples_per_class):
            image = 0.05 * torch.randn(
                IMAGE_SIZE, IMAGE_SIZE, generator=generator
            )
            if class_index == 0:  # 横线
                image[7:9, 2:14] += 1.0
            elif class_index == 1:  # 竖线
                image[2:14, 7:9] += 1.0
            else:  # 方块
                image[5:11, 5:11] += 1.0
            images.append(image.clamp(0, 1).unsqueeze(0))
            labels.append(class_index)

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)  # logits


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / total


def main():
    torch.manual_seed(0)
    images, labels = make_dataset()
    dataset = TensorDataset(images, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(1),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = SimpleCNN()
    sample_output = model(images[:4])
    print("input shape:", images[:4].shape)
    print("output shape:", sample_output.shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for batch_images, batch_labels in train_loader:
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_labels.shape[0]

        train_loss = total_loss / len(train_set)
        test_accuracy = evaluate(model, test_loader)
        print(
            f"epoch={epoch + 1:02d}, "
            f"train_loss={train_loss:.4f}, "
            f"test_accuracy={test_accuracy:.2%}"
        )

    model.eval()
    # 每一类取一张图片，方便观察三种图案分别被预测为什么类别。
    sample_images = images[[0, 200, 400]]
    with torch.no_grad():
        logits = model(sample_images)
        probabilities = torch.softmax(logits, dim=1)
        print("sample predictions:", logits.argmax(dim=1).tolist())
        print("sample probabilities:\n", probabilities)


if __name__ == "__main__":
    main()
