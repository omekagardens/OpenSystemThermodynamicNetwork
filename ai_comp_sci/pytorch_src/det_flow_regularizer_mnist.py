#!/usr/bin/env python3
"""
DET 2.0 as a flow regularizer for deep nets (PyTorch / MNIST demo)

Idea:
- Treat each layer as a node i with a scalar "free level" proxy:
    F_i ≈ E_i = mean(activation_i^2)
- Treat "flow" along the chain as these energies.
- DET-style regularizer encourages *smooth, non-vanishing, non-exploding*
  flow of energy across layers by penalizing large mismatches:

    L_det = mean_i (E_{i+1} - E_i)^2

This is a very simple, falsifiable probe: does adding L_det improve
test accuracy / stability versus baseline CE-only training?
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ------------------------------
# 1. Model: small CNN with hooks for activations
# ------------------------------

class DETConvNet(nn.Module):
    """
    Simple CNN that returns logits AND a list of intermediate activations
    so we can compute DET flow statistics.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)                            # 28x28 -> 14x14

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, return_activations: bool = False):
        activations = []

        # Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        activations.append(x)

        # Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        activations.append(x)

        x = self.pool(x)  # pooling is part of the flow path but we usually
                          # don't treat it as a separate "node" here.

        # Flatten
        x = torch.flatten(x, 1)

        # FC 1
        x = self.fc1(x)
        x = F.relu(x)
        activations.append(x)

        # FC 2 (logits only, no activation)
        x = self.fc2(x)

        if return_activations:
            return x, activations
        else:
            return x


# ------------------------------
# 2. DET flow regularizer
# ------------------------------

def det_flow_regularizer(activations, eps: float = 1e-8):
    """
    Compute a DET-style flow regularizer from a list of activations.

    Mapping to DET 2.0 (simplified probe):
    - Node i: layer i
    - Free level F_i ~ E_i = mean(activation_i^2)
    - Flux/flow along chain ~ E_i
    - Regularizer encourages smoothness of this flow:

        L_det = mean_i (E_{i+1} - E_i)^2

    This is a *very* simple, falsifiable mapping; we can refine later.
    """

    if len(activations) < 2:
        return torch.tensor(0.0, device=activations[0].device)

    # Energy per layer (proxy for free level / flux)
    energies = []
    for a in activations:
        # mean squared activation as a scalar per batch
        e = (a ** 2).mean(dim=list(range(1, a.ndim)))  # mean over non-batch dims
        e = e.mean()  # then mean over batch -> single scalar
        energies.append(e)

    energies = torch.stack(energies)  # shape [L]

    # Flow mismatch across adjacent layers
    diffs = energies[1:] - energies[:-1]
    det_loss = (diffs ** 2).mean()

    # Small epsilon floor so it's always non-zero tensor (if needed)
    return det_loss + 0.0 * eps


# ------------------------------
# 3. Training & evaluation loops
# ------------------------------

def train_epoch(model,
                device,
                train_loader,
                optimizer,
                lambda_det: float = 0.0):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_det = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        logits, activations = model(data, return_activations=True)
        ce_loss = F.cross_entropy(logits, target)

        if lambda_det > 0.0:
            det_loss = det_flow_regularizer(activations)
            loss = ce_loss + lambda_det * det_loss
        else:
            det_loss = torch.tensor(0.0, device=device)
            loss = ce_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_ce += ce_loss.item() * data.size(0)
        total_det += det_loss.item() * data.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += data.size(0)

    avg_loss = total_loss / total
    avg_ce = total_ce / total
    avg_det = total_det / total
    acc = correct / total

    return {
        "loss": avg_loss,
        "ce_loss": avg_ce,
        "det_loss": avg_det,
        "acc": acc,
    }


def eval_epoch(model, device, test_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            ce_loss = F.cross_entropy(logits, target)

            total_loss += ce_loss.item() * data.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += data.size(0)

    return {
        "loss": total_loss / total,
        "acc": correct / total,
    }


# ------------------------------
# 4. Main: CLI + run experiment
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DET 2.0 flow-regularizer probe on MNIST (PyTorch)"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-det", type=float, default=0.0,
                        help="DET regularizer strength (0.0 = baseline)")
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    # Model + optimizer
    model = DETConvNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"lambda_det = {args.lambda_det}")
    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            lambda_det=args.lambda_det,
        )
        test_stats = eval_epoch(model, device, test_loader)
        dt = time.time() - t0

        print(f"EPOCH {epoch}")
        print(f"  train loss: {train_stats['loss']:.4f} "
              f"(CE {train_stats['ce_loss']:.4f}, DET {train_stats['det_loss']:.4f})")
        print(f"  train acc:  {train_stats['acc']:.4f}")
        print(f"  test  loss: {test_stats['loss']:.4f}")
        print(f"  test  acc:  {test_stats['acc']:.4f}")
        print(f"  epoch time: {dt:.2f}s\n")


if __name__ == "__main__":
    main()