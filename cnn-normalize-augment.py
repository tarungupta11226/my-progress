import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

digits = load_digits()
X = (digits.images.astype(np.float32) / 16.0)[:, None, :, :]
y = digits.target.astype(np.int64)

perm = np.random.permutation(len(X))
n_train, n_val = int(0.6 * len(X)), int(0.2 * len(X))
train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

train_mean = float(X[train_idx].mean())
train_std = float(X[train_idx].std())
print(f"train mean: {train_mean:.4f}, train_std: {train_std:.4f}")

def random_shift(x, max_shift=1):
    C, H, W = x.shape
    padded = F.pad(x, (max_shift, max_shift, max_shift, max_shift))
    top = torch.randint(0, 2 * max_shift + 1, (1,)).item()
    left = torch.randint(0, 2 * max_shift + 1, (1,)).item()
    return padded[:, top:top + H, left:left + W]

class DigitDataset(Dataset):
    def __init__(self, images, labels, mean, std, augment=False, max_shift=1):
        self.images = torch.from_numpy(images)
        self.labels = torch.from_numpy(labels)
        self.mean, self.std = mean, std
        self.augment, self.max_shift = augment, max_shift

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = self.images[i]
        if self.augment:
            x = random_shift(x, self.max_shift)
        x = (x - self.mean) / self.std
        return x, self.labels[i]

class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n

def run_experiment(augment, epochs=30, seed=SEED):
    torch.manual_seed(seed)

    train_ds = DigitDataset(X[train_idx], y[train_idx], train_mean, train_std, augment=augment)
    val_ds = DigitDataset(X[val_idx], y[val_idx], train_mean, train_std, augment=False)
    test_ds = DigitDataset(X[val_idx], y[val_idx], train_mean, train_std, augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc, best_state, best_tr_acc = 0.0, None, 0.0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        if va_acc > best_val_acc:
            best_val_acc, best_tr_acc = va_acc, tr_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"[aug = {augment}] epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc} | val loss {va_loss:.4f} acc {va_acc:.3f}")

    model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    return {"train_acc": best_tr_acc, "val_acc": best_val_acc, "test_acc": te_acc}

results = {}
for aug in (False, True):
    results[aug] = run_experiment(augment=aug)

print("\nAugmentation | train acc | val acc | test acc | train-val gap")
for aug, r in results.items():
    print(f"{str(aug):<12} | {r['train_acc']:.3f}     | {r['val_acc']:.3f}   | {r['test_acc']:.3f}    | {r['train_acc'] - r['val_acc']:+.3f}")  
