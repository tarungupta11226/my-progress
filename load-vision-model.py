import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

SEED = 42
NUM_CLASSES = 10
IMG_SIZE = 128
BATCH_SIZE = 64
N_TRAIN = 5000
N_VAL = 1000
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
EPOCHS_HEAD = 3
EPOCHS_FT = 3

torch.manual_seed(SEED)
device = ("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
full_train = datasets.CIFAR10("./data", train=True, download=True, transform=tf)
perm = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(SEED)).tolist()
train_ds = Subset(full_train, perm[:N_TRAIN])
val_ds = Subset(full_train, perm[N_TRAIN:N_TRAIN + N_VAL])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print("Original head:", model.fc)

model.requires_grad(False)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)
print("New head:", model.fc)

def report(model, title):
    print(f"\n=== {title} ===")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        t = sum(p.numel() for p in child.parameters() if p.requires_grad)
        status = "no params" if n == 0 else ("TRAINED" if t == n else ("FROZEN" if t == 0 else "PARTIAL"))
        print(f"{name:<8} params {n:>10,} | trainable {t:>10,} | {status}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TOTAL {total:,} | trainable {trainable:,} ({100 * trainable / total:.2f}%)")
    print("Trainable tensors:", [n for n, p in model.named_parameters() if p.requires_grad][:6], "...")

report(model, "Stage 1: head only")

def set_train_mode(model):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()

def train_one_epoch(model, loader, criterion, optimizer, device):
    set_train_mode(model)
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
        xb, yb = xb.to(device),yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += xb.size(0)
    return total_loss / n, correct / n

criterion = nn.CrossEntropyLoss()

conv1_before = model.conv1.weight.detach().clone()
bn1_mean_before = model.bn1.running_mean.detach().clone()
fc_before = model.fc.weight.detach().clone()

opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD)
for epoch in range(1, EPOCHS_HEAD + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt1, device)
    va_loss, va_acc = evaluate(model, val_loader, criterion, device)
    print(f"[stage 1] epoch {epoch} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

print("\nconv1 weights unchanged: ", torch.equal(conv1_before, model.conv1.weight.detach()))
print("bn1 running_mean unchanged:", torch.equal(bn1_mean_before, model.bn1.running_mean.detach()))
print("fc weights changed: ", not torch.equal(fc_before, model.fc.weight.detach()))

for p in model.layer4.parameters():
    p.requires_grad = True
report(model, "Stage 2: layer4 + head")

layer4_before = model.layer4[0].conv1.weight.detach().clone()

opt2 = torch.optim.Adam([
    {"params": list(model.layer4.parameters()), "lr": LR_BACKBONE},
    {"params": list(model.fc.parameters()), "lr": LR_HEAD},
])
for epoch in range(1, EPOCHS_FT + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, opt2, device)
    va_loss, va_acc = evaluate(model, val_loader, criterion, device)
    print(f"[stage 2] epoch {epoch} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {va_loss:.4f} acc {va_acc:.3f}")

print("\nlayer4 weights changed: ", not torch.equal(layer4_before, model.layer4[0].conv1.weight.detach()))
print("conv1 still unchanged: ", torch.equal(conv1_before, model.conv1.weight.detach()))