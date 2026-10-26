import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets

SEED = 42
BATCH_SIZE = 138
LR = 1e-3
EPOCHS = 15
VAL_SIZE = 6000
CONV1_CH = 16
CONV2_CH = 32
HIDDEN = 128
CLASS_NAMES = ["T-shit/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_raw = datasets.FashionMNIST(root="./data", train=True, download=True)
test_raw = datasets.FashionMNIST(root="./data", train=False, download=True)

X_all = train_raw.data.float().div(255.0).unsqueeze(1)
y_all = train_raw.targets
X_test = test_raw.data.float().div(255.0).unsqueeze(1)
y_test = test_raw.targets

perm = torch.randperm(len(X_all), generator=torch.Generator().manual_seed(SEED))
val_idx, train_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]

mean = X_all[train_idx].mean().item()
std = X_all[train_idx].std().item()
normalize = lambda x: (x - mean) / std

X_train, y_train = normalize(X_all[train_idx]), y_all[train_idx]
X_val, y_val = normalize(X_all[val_idx]), y_all[val_idx]
X_test = normalize(X_test)
print(f"train {tuple(X_train.shape)}) | val {tuple(X_val.shape)} | test {tuple(X_test.shape)} mean {mean:.4f} stf {std:.4f}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

class FashionCNN(nn.Module):
    def __init__(self, c1=CONV1_CH, c2=CONV2_CH, hidden=HIDDEN, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = FashionCNN().to(device)

out = torch.zeros(4, 1, 28, 28, device=device)
for layer in list(model.features) + list(model.classifier):
    out = layer(out)
    print(f"{layer.__class__.__name__:<10} -> {tuple(out.shape)}")
assert out.shape == (4, 10)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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

@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    labels, preds = [], []
    for xb, yb in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(yb)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc, best_epoch, best_state = 0.0, 0, None

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    va_loss, va_acc = evaluate(model, val_loader, criterion, device)
    for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc)):
        history[k].append(v)
    if va_acc > best_val_acc:
        best_val_acc, best_epoch = va_acc, epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

epochs = range(1, EPOCHS + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(epochs, history["train_loss"], label="train"); ax[0].plot(epochs, history["val_loss"], label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(epochs, history["train_acc"], label="train"); ax[1].plot(epochs, history["val_acc"], label="val")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
plt.tight_layout(); plt.show()

model.load_state_dict(best_state)

def analyze_weakness(labels, preds, names):
    n = len(names)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (labels, preds), 1)

    per_class = cm.diagonal() / cm.sum(axis=1)
    order = np.argsort(per_class)
    off = cm.copy(); np.fill_diagonal(off, 0)
    i, j = np.unravel_index(off.argmax(), off.shape)
    return cm, per_class, order, (i, j, off[i, j])

val_labels, val_preds = predict_all(model, val_loader, device)
cm, per_class, order, (ti, tj, count) = analyze_weakness(val_labels, val_preds, CLASS_NAMES)
 
print(f"\nBest epoch {best_epoch}: train acc {history['train_acc'][best_epoch-1]:.4f}, val acc {best_val_acc:.4f}, "
      f"gap {history['train_acc'][best_epoch-1] - best_val_acc:+.4f}")
print("Lowest val loss at epoch", int(np.argmin(history["val_loss"])) + 1, "of", EPOCHS)
print("\nPer-class validation accuracy (worst first):")
for k in order:
    print(f"  {CLASS_NAMES[k]:<12} {per_class[k]:.3f}")
print(f"\nWEAKNESS: '{CLASS_NAMES[ti]}' is most often predicted as '{CLASS_NAMES[tj]}' ({count} validation images)")
 
te_loss, te_acc = evaluate(model, test_loader, criterion, device)
print(f"\nTest loss {te_loss:.4f} | Test acc {te_acc:.4f}")
