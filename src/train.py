import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import FashionCNN, count_params


BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
TOTAL_SAMPLES = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = Path('models')
REPORT_DIR = Path('reports')
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    full_train = datasets.FashionMNIST(
        root='data', train=True, download=True, transform=transform
    )
    full_test = datasets.FashionMNIST(
        root='data', train=False, download=True, transform=transform
    )

    classes = full_train.classes
    samples_per_class = TOTAL_SAMPLES // len(classes)

    train_indices = []
    for c in range(len(classes)):
        targets = torch.as_tensor(full_train.targets)
        pos = torch.where(targets == c)[0][:samples_per_class]
        train_indices.extend(pos.tolist())

    train_dataset = Subset(full_train, train_indices)
    test_dataset = full_test

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'Train: {len(train_dataset)} images ({samples_per_class} per class)')
    print(f'Test:  {len(test_dataset)} images')
    print(f'Classes ({len(classes)}): {classes}')

    return train_loader, test_loader, classes


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def plot_history(train_losses, train_accs, val_losses, val_accs):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(range(1, EPOCHS + 1), train_losses, 'b-', label='Train Loss')
    ax1.plot(range(1, EPOCHS + 1), val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss History')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, EPOCHS + 1), train_accs, 'b-', label='Train Acc')
    ax2.plot(range(1, EPOCHS + 1), val_accs, 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy History')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / 'training_history.png', dpi=150)
    plt.close()
    print(f'Plot saved: {REPORT_DIR / "training_history.png"}')


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Device: {DEVICE}')
    print(f'Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}, LR: {LR}')
    print(f'Total training samples: {TOTAL_SAMPLES}')
    print('-' * 50)

    train_loader, test_loader, classes = load_data()
    print('-' * 50)

    model = FashionCNN(num_classes=10).to(DEVICE)
    params = count_params(model)
    print(f'Model: {model.__class__.__name__}')
    print(f'Params: {params:,}')
    print('-' * 50)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch:2d}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / 'fashion_best.pt')
            print(f'  >> New best model saved ({best_acc:.2f}%)')

    torch.save(model.state_dict(), MODEL_DIR / 'fashion_final.pt')
    print('-' * 50)
    print(f'Best test accuracy: {best_acc:.2f}%')
    print(f'Final model saved: {MODEL_DIR / "fashion_final.pt"}')
    print(f'Best model saved:  {MODEL_DIR / "fashion_best.pt"}')

    plot_history(train_losses, train_accs, val_losses, val_accs)


if __name__ == '__main__':
    main()
