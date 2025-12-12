import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchvision import datasets, transforms

# -------------------------
# Random seed helper
# -------------------------
import random

def set_global_seed(seed: int):
    """
    Set random seeds for reproducibility across numpy, torch, and python's random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For determinism (may have some perf impact)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
      - val_var: np.array(D,)   -- variance of output unit on validation
    """
    H_train = H_train.float()
    H_val = H_val.float()

    N_train, D = H_train.shape
    N_val, D_val = H_val.shape
    assert D == D_val

    # Relational weight matrix R where R[k, j] ~= R_{j->k}
    R = torch.zeros(D, D)

    # Precompute average activation magnitude A_j on training set
    A = H_train.abs().mean(dim=0)  # (D,)

    lambdas = []
    recon_mse = []
    val_var = []

    for k in range(D):
        # Mask out unit k (we don't let it predict itself)
        mask = torch.ones(D, dtype=torch.bool)
        mask[k] = False

        X_train = H_train[:, mask]  # (N_train, D-1)
        y_train = H_train[:, k]     # (N_train,)

        X_val = H_val[:, mask]      # (N_val, D-1)
        y_val = H_val[:, k]         # (N_val,)

        var_k = y_val.var(unbiased=False).item()

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

        # Store weights into dense row k of R (zero on self-index)
        row = torch.zeros(D)
        row[mask] = w
        R[k] = row

        # Reconstruction on validation set
        y_val_hat = X_val @ w  # (N_val,)

        mse_k = F.mse_loss(y_val_hat, y_val).item()
        recon_mse.append(mse_k)

        val_var.append(var_k)

        # Relational anchor: sum_j |R_{j->k}| * A_j
        # Need A_j only for j where mask == True
        A_masked = A[mask]  # (M,)
        lambda_k = (w.abs() * A_masked).sum().item()
        lambdas.append(lambda_k)

    lambdas = np.array(lambdas)
    recon_mse = np.array(recon_mse)
    val_var = np.array(val_var)
    R = R.numpy()

    return lambdas, recon_mse, val_var, R


def summarize_results(lambdas, recon_mse, val_var, mse_eps=1e-8, var_eps=1e-8):
    """
    Print some summary stats and correlation between Λ_k and reconstruction quality,
    ignoring 'dead' units that never activate (and thus have trivial MSE=0).
    """

    # Reconstruction *quality* is inversely related to normalized MSE
    # normalized_mse_k = recon_mse_k / Var(y_val_k)
    eps = 1e-8
    recon_mse = np.array(recon_mse)
    val_var = np.array(val_var)

    # Avoid divide-by-zero: clamp very small variances
    safe_var = np.clip(val_var, var_eps, None)
    norm_mse = recon_mse / safe_var
    quality = 1.0 / (norm_mse + eps)

    # Mask out units that are trivially reconstructed or have negligible variance
    active_mask = (norm_mse > mse_eps) & (val_var > var_eps)

    num_total = len(lambdas)
    num_active = active_mask.sum()
    num_dead = num_total - num_active

    lambdas_active = lambdas[active_mask]
    recon_mse_active = recon_mse[active_mask]
    quality_active = quality[active_mask]

    print("\n=== Relational Resurrection Analysis ===")
    print(f"Num units total D = {num_total}")
    print(f"Num active (used in stats) = {num_active}")
    print(f"Num dead/trivial units    = {num_dead}")

    if num_active == 0:
        print("No active units to analyze (all trivial).")
        print("======================================\n")
        return

    print(f"Λ (anchor) stats (active only): "
          f"min={lambdas_active.min():.4f}, max={lambdas_active.max():.4f}, "
          f"mean={lambdas_active.mean():.4f}")
    norm_mse_active = norm_mse[active_mask]
    print(f"Recon MSE stats (active only): "
          f"min={recon_mse_active.min():.6f}, max={recon_mse_active.max():.6f}, "
          f"mean={recon_mse_active.mean():.6f}")
    print(f"Normalized MSE stats (active only): "
          f"min={norm_mse_active.min():.6f}, max={norm_mse_active.max():.6f}, "
          f"mean={norm_mse_active.mean():.6f}")

    # Pearson correlation between Λ_k and quality (active only)
    corr = np.corrcoef(lambdas_active, quality_active)[0, 1]
    print(f"Correlation(Λ_k, reconstruction quality) = {corr:.4f}")

    # Show top/bottom few units by Λ or quality (active)
    top_k = min(5, num_active)

    idx_active = np.where(active_mask)[0]

    # sort indices within active set
    order_anchor = np.argsort(-lambdas_active)      # descending by Λ
    order_quality = np.argsort(-quality_active)     # descending by quality

    print("\nTop units by Λ_k (anchor strength, active only):")
    for i in range(top_k):
        local_idx = order_anchor[i]
        idx = idx_active[local_idx]
        print(f"  unit {idx:3d}: Λ={lambdas[idx]:.4f}, MSE={recon_mse[idx]:.6f}, normMSE={norm_mse[idx]:.6f}")

    print("\nTop units by reconstruction quality (1/MSE, active only):")
    for i in range(top_k):
        local_idx = order_quality[i]
        idx = idx_active[local_idx]
        print(f"  unit {idx:3d}: Λ={lambdas[idx]:.4f}, MSE={recon_mse[idx]:.6f}, normMSE={norm_mse[idx]:.6f}")

    print("======================================\n")


def eval_with_unit_ablation(model, loader, device, unit_idx, R=None, mode="delete"):
    """
    Evaluate the model on a loader while:
      - 'delete': zeroing out hidden unit `unit_idx`
      - 'resurrect': zeroing it out, then reconstructing it from other units using R.
    """
    model.eval()
    total_correct = 0
    total_examples = 0

    # Convert R row for this unit to a torch tensor on the correct device, if needed
    R_row = None
    if mode == "resurrect":
        if R is None:
            raise ValueError("R must be provided for mode='resurrect'")
        R_row = torch.from_numpy(R[unit_idx]).to(device)  # shape (D,)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Manual forward: fc1 + ReLU, then apply ablation / resurrection on h
            x_flat = x.view(x.size(0), -1)
            h = F.relu(model.fc1(x_flat))  # (B, D)

            if mode == "delete":
                h[:, unit_idx] = 0.0
            elif mode == "resurrect":
                # First ablate, then reconstruct using relational weights
                h[:, unit_idx] = 0.0
                # Reconstruction: h_k_hat = sum_j R[k, j] * h_j
                h[:, unit_idx] = h @ R_row

            logits = model.fc2(h)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_examples += x.size(0)

    return total_correct / total_examples

# ============================================================
# Multi-unit ablation + resurrection experiment helpers
# ============================================================
def eval_with_group_ablation(model, loader, device, unit_indices, R=None, mode="delete"):
    """
    Evaluate the model on a loader while:
      - 'delete': zeroing out a group of hidden units in `unit_indices`
      - 'resurrect': zeroing them, then reconstructing them jointly from other units using R.
    """
    model.eval()
    total_correct = 0
    total_examples = 0

    # Normalize indices to a NumPy array for indexing
    unit_indices = np.array(unit_indices, dtype=int)

    R_sub = None
    if mode == "resurrect":
        if R is None:
            raise ValueError("R must be provided for mode='resurrect'")
        # R_sub has shape (K, D), where each row is R[k, :]
        R_sub = torch.from_numpy(R[unit_indices]).to(device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Manual forward: fc1 + ReLU, then apply ablation / resurrection on h
            x_flat = x.view(x.size(0), -1)
            h = F.relu(model.fc1(x_flat))  # (B, D)

            if mode == "delete":
                h[:, unit_indices] = 0.0
            elif mode == "resurrect":
                # First ablate all in the group
                h[:, unit_indices] = 0.0
                # Reconstruction: H_rec (B, K) = h (B, D) @ R_sub^T (D, K)
                H_rec = h @ R_sub.t()
                h[:, unit_indices] = H_rec

            logits = model.fc2(h)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_examples += x.size(0)

    return total_correct / total_examples


def run_delete_and_resurrect_experiment(model, val_loader, device, R, recon_mse, val_var,
                                        K=5, mse_eps=1e-8, var_eps=1e-8):
    """
    Compare accuracy drop and recovery for:
      - K most resurrectable units (lowest normalized MSE)
      - K least resurrectable units (highest normalized MSE)
    using single-unit ablation and resurrection.
    """
    # Baseline accuracy on validation set
    _, base_acc = eval_epoch(model, val_loader, device)

    recon_mse = np.array(recon_mse)
    val_var = np.array(val_var)

    # Normalized MSE and quality
    eps = 1e-8
    safe_var = np.clip(val_var, var_eps, None)
    norm_mse = recon_mse / safe_var
    quality = 1.0 / (norm_mse + eps)

    # Active units only
    active_mask = (norm_mse > mse_eps) & (val_var > var_eps)
    idx_active = np.where(active_mask)[0]

    if len(idx_active) == 0:
        print("No active units available for delete+resurrect experiment.")
        return

    # Sort active units by normalized MSE (ascending = most resurrectable)
    norm_mse_active = norm_mse[active_mask]
    order = np.argsort(norm_mse_active)  # ascending
    K_eff = min(K, len(order))

    best_indices = idx_active[order[:K_eff]]
    worst_indices = idx_active[order[-K_eff:]]

    def group_stats(indices, label):
        drops = []
        recovers = []
        print(f"\n[{label}] units: {list(indices)}")
        for k in indices:
            acc_del = eval_with_unit_ablation(model, val_loader, device, unit_idx=k, mode="delete")
            acc_res = eval_with_unit_ablation(model, val_loader, device, unit_idx=k, R=R, mode="resurrect")
            drop = base_acc - acc_del
            recover = acc_res - acc_del
            drops.append(drop)
            recovers.append(recover)
            print(f"  unit {k:3d}: base_acc={base_acc:.4f}, "
                  f"acc_del={acc_del:.4f}, acc_res={acc_res:.4f}, "
                  f"drop={drop:.4f}, recover={recover:.4f}")
        print(f"  {label} mean drop:    {np.mean(drops):.4f}")
        print(f"  {label} mean recover: {np.mean(recovers):.4f}")

    print("\n=== Delete + Resurrect Experiment (single-unit) ===")
    print(f"Baseline val accuracy: {base_acc:.4f}")
    group_stats(best_indices, "Most resurrectable (lowest normMSE)")
    group_stats(worst_indices, "Least resurrectable (highest normMSE)")
    print("======================================\n")

# ============================================================
# Multi-unit ablation + resurrection experiment
# ============================================================
def run_multiunit_ablation_experiment(model, val_loader, device, R, recon_mse, val_var,
                                      group_size=10, mse_eps=1e-8, var_eps=1e-8):
    """
    Compare accuracy drop and recovery when ablating *groups* of units:
      - group_size most resurrectable units (lowest normalized MSE)
      - group_size least resurrectable units (highest normalized MSE)
      - group_size random active units (control)
    """
    # Baseline accuracy on validation set
    _, base_acc = eval_epoch(model, val_loader, device)

    recon_mse = np.array(recon_mse)
    val_var = np.array(val_var)

    # Normalized MSE and quality
    eps = 1e-8
    safe_var = np.clip(val_var, var_eps, None)
    norm_mse = recon_mse / safe_var
    quality = 1.0 / (norm_mse + eps)

    # Active units only
    active_mask = (norm_mse > mse_eps) & (val_var > var_eps)
    idx_active = np.where(active_mask)[0]

    if len(idx_active) == 0:
        print("No active units available for multi-unit ablation experiment.")
        return

    norm_mse_active = norm_mse[active_mask]
    order = np.argsort(norm_mse_active)  # ascending = most resurrectable
    K_eff = min(group_size, len(order))

    best_indices = idx_active[order[:K_eff]]
    worst_indices = idx_active[order[-K_eff:]]

    # Random control group from active units
    rng = np.random.default_rng()
    rand_indices = rng.choice(idx_active, size=K_eff, replace=False)

    def group_eval(indices, label):
        acc_del = eval_with_group_ablation(model, val_loader, device, indices, mode="delete")
        acc_res = eval_with_group_ablation(model, val_loader, device, indices, R=R, mode="resurrect")
        drop = base_acc - acc_del
        recover = acc_res - acc_del
        print(f"\n[{label}] units: {list(indices)}")
        print(f"  base_acc={base_acc:.4f}, acc_del={acc_del:.4f}, acc_res={acc_res:.4f}, "
              f"drop={drop:.44f}, recover={recover:.4f}")
        return drop, recover

    print("\n=== Multi-Unit Delete + Resurrect Experiment ===")
    print(f"Baseline val accuracy: {base_acc:.4f}")

    best_drop, best_rec = group_eval(best_indices, "Most resurrectable (lowest normMSE)")
    worst_drop, worst_rec = group_eval(worst_indices, "Least resurrectable (highest normMSE)")
    rand_drop, rand_rec = group_eval(rand_indices, "Random active")

    print("\nSummary (group_size = %d):" % K_eff)
    print(f"  Most resurrectable:  drop={best_drop:.4f}, recover={best_rec:.4f}")
    print(f"  Least resurrectable: drop={worst_drop:.4f}, recover={worst_rec:.4f}")
    print(f"  Random:              drop={rand_drop:.4f}, recover={rand_rec:.4f}")
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)
    print(f"Using seed: {args.seed}")

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
    lambdas, recon_mse, val_var, R = compute_relational_matrix_and_anchor(
        H_train, H_val, ridge_lambda=args.ridge_lambda
    )

    summarize_results(lambdas, recon_mse, val_var)

    # --------------------------
    # Delete + resurrect experiment (single-unit)
    # --------------------------
    run_delete_and_resurrect_experiment(
        model, val_loader, device, R, recon_mse, val_var, K=5
    )

    # --------------------------
    # Multi-unit delete + resurrect experiment
    # --------------------------
    run_multiunit_ablation_experiment(
        model, val_loader, device, R, recon_mse, val_var, group_size=10
    )


if __name__ == "__main__":
    main()