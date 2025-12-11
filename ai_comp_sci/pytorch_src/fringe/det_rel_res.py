import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# =========================
#  Model definition
# =========================

class SimpleMLP(nn.Module):
    """
    Simple 1-hidden-layer MLP for MNIST.
    We'll treat the hidden units as our "persons" / nodes.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.hidden_dim = hidden_dim

    def forward(self, x, return_hidden=False):
        # x: (B, 1, 28, 28)
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc1(x))  # hidden activations
        out = self.fc2(h)
        if return_hidden:
            return out, h
        return out


# =========================
#  Training utilities
# =========================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_examples += x.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_examples += x.size(0)

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


# =========================
#  Activation capture
# =========================

def collect_hidden_activations(model, loader, device, max_batches=None):
    """
    Run the model over a loader and collect all hidden activations
    into a single tensor H of shape (N, D), where:
      - N = number of examples (possibly truncated by max_batches)
      - D = hidden_dim
    """
    model.eval()
    activations = []

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device)
            _, h = model(x, return_hidden=True)
            activations.append(h.cpu())

    H = torch.cat(activations, dim=0)  # (N, D)
    return H


# =========================
#  Relational reconstruction
# =========================

def compute_relational_matrix_and_anchor(H_train, H_val, ridge_lambda=1e-3):
    """
    For each hidden unit k, fit a ridge regression:
        h_k ≈ sum_{j != k} R_{j->k} h_j

    Using training activations H_train (N_train, D).
    Evaluate reconstruction on validation activations H_val (N_val, D).

    Returns:
      - lambdas: np.array(D,)  -- relational anchor Λ_k
      - recon_mse: np.array(D,) -- MSE reconstruction error on validation
    """
    H_train = H_train.float()
    H_val = H_val.float()

    N_train, D = H_train.shape
    N_val, D_val = H_val.shape
    assert D == D_val

    # Precompute average activation magnitude A_j on training set
    A = H_train.abs().mean(dim=0)  # (D,)

    lambdas = []
    recon_mse = []

    for k in range(D):
        # Mask out unit k (we don't let it predict itself)
        mask = torch.ones(D, dtype=torch.bool)
        mask[k] = False

        X_train = H_train[:, mask]  # (N_train, D-1)
        y_train = H_train[:, k]     # (N_train,)

        X_val = H_val[:, mask]      # (N_val, D-1)
        y_val = H_val[:, k]         # (N_val,)

        # Solve ridge: w = (X^T X + λI)^(-1) X^T y
        # Shapes:
        #   X_train: (N, M), M = D-1
        #   w: (M,)
        M = X_train.shape[1]
        XtX = X_train.t() @ X_train      # (M, M)
        reg = ridge_lambda * torch.eye(M)
        XtX_reg = XtX + reg

        Xty = X_train.t() @ y_train      # (M,)

        w = torch.linalg.solve(XtX_reg, Xty)  # (M,)

        # Reconstruction on validation set
        y_val_hat = X_val @ w  # (N_val,)

        mse_k = F.mse_loss(y_val_hat, y_val).item()
        recon_mse.append(mse_k)

        # Relational anchor: sum_j |R_{j->k}| * A_j
        # Need A_j only for j where mask == True
        A_masked = A[mask]  # (M,)
        lambda_k = (w.abs() * A_masked).sum().item()
        lambdas.append(lambda_k)

    lambdas = np.array(lambdas)
    recon_mse = np.array(recon_mse)

    return lambdas, recon_mse


def summarize_results(lambdas, recon_mse):
    """
    Print some summary stats and correlation between Λ_k and reconstruction quality.
    Reconstruction *quality* is inversely related to MSE,
    so we look at 1 / (MSE + eps).
    """
    eps = 1e-8
    quality = 1.0 / (recon_mse + eps)

    print("\n=== Relational Resurrection Analysis ===")
    print(f"Num units D = {len(lambdas)}")
    print(f"Λ (anchor) stats: min={lambdas.min():.4f}, max={lambdas.max():.4f}, "
          f"mean={lambdas.mean():.4f}")
    print(f"Recon MSE stats: min={recon_mse.min():.6f}, max={recon_mse.max():.6f}, "
          f"mean={recon_mse.mean():.6f}")

    # Pearson correlation between Λ_k and quality
    corr = np.corrcoef(lambdas, quality)[0, 1]
    print(f"Correlation(Λ_k, reconstruction quality) = {corr:.4f}")

    # Optionally, show top/bottom few units by Λ or quality
    top_k = min(5, len(lambdas))

    idx_sorted_anchor = np.argsort(-lambdas)  # descending
    idx_sorted_quality = np.argsort(-quality)  # descending

    print("\nTop units by Λ_k (anchor strength):")
    for i in range(top_k):
        idx = idx_sorted_anchor[i]
        print(f"  unit {idx:3d}: Λ={lambdas[idx]:.4f}, MSE={recon_mse[idx]:.6f}")

    print("\nTop units by reconstruction quality (1/MSE):")
    for i in range(top_k):
        idx = idx_sorted_quality[i]
        print(f"  unit {idx:3d}: Λ={lambdas[idx]:.4f}, MSE={recon_mse[idx]:.6f}")

    print("======================================\n")


# =========================
#  Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--max-train-batches-for_H", type=int, default=200,
                        help="Limit for number of batches to collect train activations (None for all)")
    parser.add_argument("--max-val-batches-for_H", type=int, default=200,
                        help="Limit for number of batches to collect val activations (None for all)")
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------
    # Load MNIST
    # --------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)

    # Split train into train/val for reconstruction experiments
    n_total = len(full_train)
    n_val = int(0.2 * n_total)
    n_train = n_total - n_val

    train_ds, val_ds = random_split(full_train, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # --------------------------
    # Init model, optimizer
    # --------------------------
    model = SimpleMLP(hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --------------------------
    # Train
    # --------------------------
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(f"EPOCH {epoch}")
        print(f"  train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"  val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

    # --------------------------
    # Collect hidden activations
    # --------------------------
    print("\nCollecting hidden activations for relational analysis...")

    H_train = collect_hidden_activations(
        model,
        train_loader,
        device,
        max_batches=args.max_train_batches_for_H
    )
    H_val = collect_hidden_activations(
        model,
        val_loader,
        device,
        max_batches=args.max_val_batches_for_H
    )

    print(f"H_train shape: {H_train.shape}")
    print(f"H_val   shape: {H_val.shape}")

    # --------------------------
    # Compute relational anchors & reconstruction
    # --------------------------
    lambdas, recon_mse = compute_relational_matrix_and_anchor(
        H_train, H_val, ridge_lambda=args.ridge_lambda
    )

    summarize_results(lambdas, recon_mse)


if __name__ == "__main__":
    main()