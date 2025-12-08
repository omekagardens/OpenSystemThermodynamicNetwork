## DET Flow Regularizer – Probe #1 (MNIST / ConvNet)

**Goal.** Test a simple DET 2.0–inspired regularizer on a standard deep learning task and check whether it measurably changes training/generalization behavior compared to a cross-entropy–only baseline.

### Setup

- Dataset: MNIST (train 60k / test 10k, standard normalization)
- Model: small CNN
  - conv1: 1 → 32, 3×3, ReLU
  - conv2: 32 → 64, 3×3, ReLU
  - max-pool 2×2
  - fc1: 64·14·14 → 128, ReLU
  - fc2: 128 → 10 (logits)
- Optimizer: Adam (lr = 1e-3)
- Batch size: 128
- Epochs: 5
- Device: CPU
- Seed: 42

### DET mapping (simplified)

For this probe we map the DET 2.0 quantities as:

- Node free-level proxy:  
  \( F_i \approx E_i = \mathbb{E}[\text{activation}_i^2] \)  
  (mean squared activation per layer over batch and spatial/feature dims)
- Flow along the chain: the sequence of energies \(E_i\) across layers.
- Regularizer: penalize mismatches in “flow” between adjacent layers:

\[
L_{\text{DET}} = \frac{1}{L-1} \sum_{i=1}^{L-1} (E_{i+1} - E_i)^2
\]

Total loss:

\[
L = L_{\text{CE}} + \lambda_{\text{DET}} \, L_{\text{DET}}.
\]

This is deliberately simple: a scalar per layer, no explicit σ or reservoir yet. It’s just “encourage smooth activation energy flow across the depth of the network.”

### Results (single seed)

Final-epoch metrics:

| λ_det | Train acc | Test acc | Best test acc | Final CE loss | Final DET loss |
|------:|----------:|---------:|--------------:|--------------:|---------------:|
| 0.00  | 0.9959    | 0.9871   | 0.9888        | 0.0118        | 0.0000         |
| 0.05  | 0.9996    | 0.9919   | 0.9919        | 0.0055        | 0.0988         |
| 0.10  | 0.9995    | 0.9913   | 0.9913        | 0.0059        | 0.0563         |

Observations:

- Both λ_det = 0.05 and λ_det = 0.10 **improve test accuracy** over the CE-only baseline
  by ~0.25–0.5 percentage points in this configuration.
- Cross-entropy loss at convergence is also lower with DET, while test performance improves,
  suggesting the regularizer is not merely forcing underfitting.
- DET loss remains non-zero at the end of training, so the model does not trivially drive
  all energies to zero; instead it learns a compromise between task loss and smooth flow.

### Provisional claim (Probe #1)

For this architecture and hyperparameters:

> Adding a DET-style activation-flow regularizer that penalizes squared differences
> between per-layer activation energies leads to a small but consistent improvement
> in generalization on MNIST, compared to a cross-entropy–only baseline.

This claim is falsifiable by:

- Repeating the experiment with different random seeds.
- Varying λ_det over a wider range.
- Testing harder datasets and deeper architectures (e.g., CIFAR-10 with ResNets).

If DET is irrelevant, the curves for λ_det > 0 should match λ_det = 0 within noise across these variations. If the trend above persists, it supports the view that DET’s resource-flow framing can be turned into useful inductive biases for deep networks.