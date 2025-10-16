# train_rejectors_fixed_expert.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Bernoulli

# ───────────────────────────── Setup ─────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

# CIFAR-10 normalization (same as your script)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
testloader  = DataLoader(testset,  batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# ───────────────────── Architectures (unchanged) ─────────────────────
class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),    nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# MLP client
class ClientMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3*32*32, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, 512),     nn.BatchNorm1d(512),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),      nn.BatchNorm1d(256),  nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)

# 70% client
class ClientCNN_70(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# 75% client
class ClientCNN_75(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96*8*8, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# 80% client
class ClientCNN_80(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96*8*8, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# 83.3% client
class ClientCNN_83_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),     nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class Rejector(nn.Module):
    def __init__(self, hidden=256, drop=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(3 * 32 * 32, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, 3)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten to [B, 3072]
        return self.block(x)


# ─────────────────────── Losses (UNCHANGED) ───────────────────────
def surrogate_loss_rejector(r_logits, e_logits, m_logits, labels, c=0.5):
    e_pred = e_logits.argmax(dim=1)
    m_pred = m_logits.argmax(dim=1)
    correct_e = (e_pred == labels).float()
    correct_m = (m_pred == labels).float()
    logp = F.log_softmax(r_logits, dim=1)
    term1 = correct_m * logp[:, 0]
    term2 = correct_e * logp[:, 1]
    term3 = torch.bernoulli(torch.tensor(1-c)).item() * logp[:, 2]
    return -(term1 + term2 + term3).sum()

def surrogate_loss_expert(expert_outputs, true_ys_index):
    return F.cross_entropy(expert_outputs, true_ys_index, reduction='sum')

# ─────────────────────── Utilities ───────────────────────
def load_frozen(model: nn.Module, path: str):
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model

@torch.no_grad()
def eval_top1(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

# ─────────────────────── Training (Rejector only) ───────────────────────
def train_rejector_for_client(client_key, client_model, client_ckpt, expert, epochs=10, lr=1e-3):
    print(f"\n─── Training Rejector for client: {client_key} ───")

    # Load frozen client & expert
    client = client_model().to(device)
    load_frozen(client, client_ckpt)
    expert.eval()  # already frozen

    # Fresh rejector
    rejector = Rejector().to(device)
    optimizer = optim.Adam(rejector.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        rejector.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(trainloader, 1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.no_grad():
                logits_e = expert(x)   # fixed expert
                logits_m = client(x)   # fixed client

            logits_r = rejector(x)     # trainable rejector
            loss = surrogate_loss_rejector(logits_r, logits_e, logits_m, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                print(f"[{client_key}] Epoch {epoch:02d} Iter {i:04d}  Loss_r={loss.item():.2f}")

        avg_loss = running_loss / len(trainloader)
        # Optional: quick sanity metric on test set (expert top-1)
        with torch.no_grad():
            acc_e = eval_top1(expert, testloader)
            acc_c = eval_top1(client, testloader)
        print(f"[{client_key}] Epoch {epoch:02d} | Avg Rejector Loss: {avg_loss:.2f} | Expert@Top1: {acc_e:.2f}% | Client@Top1: {acc_c:.2f}%")

    # Save ONLY the rejector
    out_path = f"RejectorModels3L/rejector_{client_key}.pt"
    torch.save(rejector.state_dict(), out_path)
    print(f"✅ Saved: {out_path}")

# ─────────────────────── Main ───────────────────────
if __name__ == "__main__":
    # Load frozen Expert
    expert = load_frozen(Expert(), "expert_model.pt")

    # Map client keys → (class, ckpt path)
    clients = {
        "mlp_3l":   (ClientMLP,      "ClientModels/client_model_mlp.pt"),
        "70_3l":    (ClientCNN_70,   "ClientModels/client_model70.pt"),
        "75_3l":    (ClientCNN_75,   "ClientModels/client_model75.pt"),
        "80_3l":    (ClientCNN_80,   "ClientModels/client_model80.pt"),
        "83_3_3l":  (ClientCNN_83_3, "ClientModels/client_model83_3.pt"),
    }

    # Train a separate rejector for each client
    # Tweak epochs/lr if you like; defaults are modest for quick runs
    for key, (cls, ckpt) in clients.items():
        if not os.path.isfile(ckpt):
            print(f"⚠️ Skipping {key}: missing checkpoint {ckpt}")
            continue
        train_rejector_for_client(key, cls, ckpt, expert, epochs=10, lr=1e-3)

