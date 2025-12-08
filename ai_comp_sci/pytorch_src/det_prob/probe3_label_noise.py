import torch
import torch.nn.functional as F
from dataset import load_tiny_mnist
from det_mlp import DetMLP
from baseline_mlp import BaselineMLP

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# Config
label_noise = 0.5      # 50% of training labels corrupted
lambda_det  = 0.05
epochs      = 15

print(f"Label noise: {label_noise}, lambda_det: {lambda_det}")

train_loader, test_loader = load_tiny_mnist(label_noise=label_noise)

def eval_on_loader(model, loader, det=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).view(x.size(0), -1)
            y = y.to(device)

            if det:
                logits, _, _ = model(x)
            else:
                logits = model(x)

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# DET model
det_model = DetMLP().to(device)
opt_det = torch.optim.Adam(det_model.parameters(), lr=1e-3)

# Baseline model
base_model = BaselineMLP().to(device)
opt_base = torch.optim.Adam(base_model.parameters(), lr=1e-3)

for epoch in range(epochs):
    print(f"\nEPOCH {epoch+1}")

    # ---- DET MODEL ----
    det_model.train()
    for x, y in train_loader:
        x = x.to(device).view(x.size(0), -1)
        y = y.to(device)

        logits, det_loss, _ = det_model(x)
        ce = F.cross_entropy(logits, y)
        loss = ce + lambda_det * det_loss

        opt_det.zero_grad()
        loss.backward()
        opt_det.step()

    det_train_acc = eval_on_loader(det_model, train_loader, det=True)
    det_test_acc = eval_on_loader(det_model, test_loader, det=True)
    print(f"DET train acc:      {det_train_acc:.4f}, test acc: {det_test_acc:.4f}")

    # ---- BASELINE ----
    base_model.train()
    for x, y in train_loader:
        x = x.to(device).view(x.size(0), -1)
        y = y.to(device)

        logits = base_model(x)
        loss = F.cross_entropy(logits, y)

        opt_base.zero_grad()
        loss.backward()
        opt_base.step()

    base_train_acc = eval_on_loader(base_model, train_loader, det=False)
    base_test_acc = eval_on_loader(base_model, test_loader, det=False)
    print(f"BASELINE train acc: {base_train_acc:.4f}, test acc: {base_test_acc:.4f}")