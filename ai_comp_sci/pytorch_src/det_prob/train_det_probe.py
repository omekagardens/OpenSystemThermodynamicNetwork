import torch
import torch.nn.functional as F
from dataset import load_tiny_mnist
from det_mlp import DetMLP
from baseline_mlp import BaselineMLP

train_loader, test_loader = load_tiny_mnist()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

det_model = DetMLP().to(device)
base_model = BaselineMLP().to(device)

opt_det = torch.optim.Adam(det_model.parameters(), lr=1e-3)
opt_base = torch.optim.Adam(base_model.parameters(), lr=1e-3)

lambda_det = 1e-3  # strength of DET term

def eval_model(model, det=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
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

lambda_values = [0.0, 1e-4, 1e-3, 1e-2, 5e-2]

for lambda_det in lambda_values:
    print("\n" + "="*60)
    print(f"Running DET probe with lambda_det = {lambda_det}")
    print("="*60)

    det_model = DetMLP().to(device)
    base_model = BaselineMLP().to(device)

    opt_det = torch.optim.Adam(det_model.parameters(), lr=1e-3)
    opt_base = torch.optim.Adam(base_model.parameters(), lr=1e-3)

    for epoch in range(5):
        print(f"\nEPOCH {epoch+1}")

        # ---- DET MODEL ----
        det_model.train()

        det_epoch_losses = []
        det_epoch_G_in_l1 = []
        det_epoch_G_out_l1 = []
        det_epoch_J_res_l1 = []
        det_epoch_sigma_l1 = []

        for x, y in train_loader:
            x = x.to(device).view(x.size(0), -1)
            y = y.to(device)

            logits, det_loss, debug = det_model(x)

            ce = F.cross_entropy(logits, y)
            loss = ce + lambda_det * det_loss

            opt_det.zero_grad()
            loss.backward()
            opt_det.step()

            det_epoch_losses.append(det_loss.detach().item())
            det_epoch_G_in_l1.append(debug["l1"]["G_in"])
            det_epoch_G_out_l1.append(debug["l1"]["G_out"])
            det_epoch_J_res_l1.append(debug["l1"]["J_res"])
            det_epoch_sigma_l1.append(debug["l1"]["sigma"])

        det_acc = eval_model(det_model, det=True)
        print(f"DET accuracy: {det_acc:.4f}")
        print(
            "DET l1 avg: "
            f"G_in={sum(det_epoch_G_in_l1)/len(det_epoch_G_in_l1):.3f}, "
            f"G_out={sum(det_epoch_G_out_l1)/len(det_epoch_G_out_l1):.3f}, "
            f"J_res={sum(det_epoch_J_res_l1)/len(det_epoch_J_res_l1):.3f}, "
            f"sigma={sum(det_epoch_sigma_l1)/len(det_epoch_sigma_l1):.4f}, "
            f"det_loss={sum(det_epoch_losses)/len(det_epoch_losses):.6f}"
        )

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

        base_acc = eval_model(base_model, det=False)
        print(f"BASELINE accuracy: {base_acc:.4f}")