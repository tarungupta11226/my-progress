import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
NUM_CLASSES = 10
IMG_SIZE = 64
BATCH_SIZE = 64
N_TRAIN = 5000
N_VAL = 1000
N_TEST = 1000
EPOCHS = 6
CLASS_NAMES = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

torch.manual_seed(SEED)
device = ("cuda" if torch.cuda.is_available() else "cpu")

base = datasets.CIFAR10("./data", train=True, download=True)
perm = torch.randperm(len(base), generator=torch.Generator().manual_seed(SEED)).tolist()
train_idx = perm[:N_TRAIN]
val_idx = perm[N_TRAIN:N_TRAIN + N_VAL]
test_idx = perm[N_TRAIN+N_VAL:N_TRAIN + N_VAL + N_TEST]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def make_loaders(normalize_mean, normalize_std):
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    full = datasets.CIFAR10("./data", train=True, download=False, transform=tf)
    tr = DataLoader(Subset(full, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED))
    va = DataLoader(Subset(full, val_idx), batch_size=BATCH_SIZE, shuffle=False)
    te = DataLoader(Subset(full, test_idx), batch_size=BATCH_SIZE, shuffle=False)
    return tr, va, te

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

def run(model, train_loader, val_loader, test_loader, optimizer, epochs, device, tag):
    criterion = nn.CrossEntropyLoss()
    history = {"train_acc": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state ={k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"[{tag}] epoch {epoch} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss{va_loss:.4f} acc {va_acc:.3f}")
    model.load_state_dict(best_state)
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    return model, history, best_val_acc, te_acc

class ScratchCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

tmp_tf = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor()])
tmp_full = datasets.CIFAR10("./data", train=True, download=False, transform=tmp_tf)
tmp_loader = DataLoader(Subset(tmp_full, train_idx), batch_size=256)
s, s2, n_px = 0.0, 0.0, 0
for xb, _ in tmp_loader:
    s += xb.sum(dim=[0, 2, 3]); s2 += (xb ** 2).sum(dim=[0, 2, 3]); n_px += xb.size(0) * xb.size(2) * xb.size(3)
cifar_mean = (s / n_px).tolist()
cifar_std = ((s2 / n_px - (s / n_px) ** 2).clamp(min=1e-8).sqrt()).tolist()
print("CIFAR subset per-channel mean/std (train split only):", [round(m, 3) for m in cifar_mean], [round(s_, 3) for s_ in cifar_std])

scratch_tr, scratch_va, scratch_te = make_loaders(cifar_mean, cifar_std)
scratch_model = ScratchCNN(NUM_CLASSES).to(device)
scratch_params = sum(p.numel() for p in scratch_model.parameters() if p.requires_grad)
scratch_opt = torch.optim.Adam(scratch_model.parameters(), lr=1e-3)

scratch_model, scratch_hist, scratch_val_acc, scratch_test_acc = run(
    scratch_model, scratch_tr, scratch_va, scratch_te, scratch_opt, EPOCHS, device, "scratch")

resnet_tr, resnet_va, resnet_te = make_loaders(IMAGENET_MEAN, IMAGENET_STD)

resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.requires_grad_(False)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
for p in resnet.layer4.parameters():
    p.requires_grad = True
resnet = resnet.to(device)

resnet_trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
resnet_total = sum(p.numel() for p in resnet.parameters())
resnet_opt = torch.optim.Adam([
    {"params": resnet.layer4.parameters(), "lr": 1e-4},
    {"params": resnet.fc.parameters(), "lr": 1e-3}
])

resnet, resnet_hist, resnet_val_acc, resnet_test_acc = run(
    resnet, resnet_tr, resnet_va, resnet_te, resnet_opt, EPOCHS, device, "resnet")

print("\n=== Fairness check ===")
print("Same train indices used by both:", scratch_tr.dataset.indices == resnet_tr.dataset.indices)
print("Same val indices used by both:  ", scratch_va.dataset.indices == resnet_va.dataset.indices)
print("Same test indices used by both: ", scratch_te.dataset.indices == resnet_te.dataset.indices)
print("Same batch size:", scratch_tr.batch_size == resnet_tr.batch_size)
print("Same epoch budget:", EPOCHS, "for both")

print(f"\n{'Model':<22}{'Trainable params':<18}{'% of total':<12}{'Best val acc':<14}{'Test acc':<10}")
print(f"{'Scratch CNN':<22}{scratch_params:<18,}{100.0:<12.2f}{scratch_val_acc:<14.3f}{scratch_test_acc:<10.3f}")
print(f"{'Fine-tuned ResNet18':<22}{resnet_trainable:<18,}{100*resnet_trainable/resnet_total:<12.2f}{resnet_val_acc:<14.3f}{resnet_test_acc:<10.3f}")