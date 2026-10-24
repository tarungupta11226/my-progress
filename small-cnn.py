import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

digits = load_digits()
X = digits.images.astype(np.float32) / 16.0
y = digits.target.astype(np.int64)
X = X[:, None, :, :]

perm = np.random.permutation(len(X))
n_train, n_val = int(0.6 * len(X)), int(0.2 * len(X))
train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

def make_loader(idx, shuffle, batch_size=64):
    ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(y[idx]))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(train_idx, shuffle=True)
val_loader = make_loader(val_idx, shuffle=False)
test_loader = make_loader(test_idx, shuffle=False)

class smallCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = smallCNN().to(device)

out = torch.zeros(4, 1, 8, 8, device=device)
for layer in list(model.features) + list(model.classifier):
    out = layer(out)
    print(f"{layer.__class__.__name__:<10} -> {tuple(out.shape)}")
assert out.shape == (4, 10), f"unexpected out shape {out.shape}"
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

EPOCH = 30
best_val_acc, best_state = 0.0, None

for epoch in range(1, EPOCH + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    va_loss, va_acc = evaluate(model, val_loader, criterion, device)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

model.load_state_dict(best_state)
te_loss, te_acc = evaluate(model, test_loader, criterion, device)
print(f"\nBest val acc {best_val_acc:.3f} | Test loss {te_loss:.4f} | Test acc {te_acc:.3f}")