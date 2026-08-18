import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


IMAGE_SIZE = 16
CLASS_COUNT = 3


def make_dataset(samples_per_class=180):
    """生成横线、竖线、方块三类带噪声的图片。"""
    generator = torch.Generator().manual_seed(11)
    images = []
    labels = []

    for class_index in range(CLASS_COUNT):
        for _ in range(samples_per_class):
            image = 0.08 * torch.randn(
                IMAGE_SIZE, IMAGE_SIZE, generator=generator
            )
            if class_index == 0:
                image[7:9, 2:14] += 1.0  # 横线
            elif class_index == 1:
                image[2:14, 7:9] += 1.0  # 竖线
            else:
                image[5:11, 5:11] += 1.0  # 方块
            images.append(image.clamp(0, 1).unsqueeze(0))
            labels.append(class_index)

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + x)


class AdvancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResidualBlock(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(32, CLASS_COUNT),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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
    torch.manual_seed(11)
    images, labels = make_dataset()
    dataset = TensorDataset(images, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(12),
    )
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    model = AdvancedCNN()
    print("model output shape:", model(images[:4]).shape)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    for epoch in range(12):
        model.train()
        total_loss = 0.0
        for batch_images, batch_labels in train_loader:
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_labels.shape[0]

        scheduler.step()
        train_loss = total_loss / len(train_set)
        test_accuracy = evaluate(model, test_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch={epoch + 1:02d}, loss={train_loss:.4f}, "
            f"test_accuracy={test_accuracy:.2%}, lr={current_lr:.5f}"
        )

    # 三类各取一张图片，查看预测类别和概率。
    model.eval()
    sample_images = images[[0, 180, 360]]
    with torch.no_grad():
        logits = model(sample_images)
        probabilities = torch.softmax(logits, dim=1)
    print("sample predictions:", logits.argmax(dim=1).tolist())
    print("sample probabilities:\n", probabilities)


if __name__ == "__main__":
    main()
