"""
PyTorch DET 2.0 Demo: Reducing Reward Hacking in a 2-Armed Bandit

This is the same conceptual experiment as before, but implemented with PyTorch:

- We have a 2-armed bandit:
    Arm 0: "aligned"
        local_reward ~ N(1.0, 0.1)
        global_benefit = +1.0   (good for the system)
    Arm 1: "hack"
        local_reward ~ N(2.0, 0.1)
        global_benefit = -1.0   (bad for the system)

- We define two agents, each with a tiny PyTorch "model" of Q-values:

    1) BaselineTorchAgent:
       - Q-values are nn.Parameter([q0, q1]).
       - Uses MSE to fit Q[action] to local_reward.

    2) DETTorchAgent:
       - Also has nn.Parameter([q0, q1]).
       - Maintains DET-style potentials F[action] (no grad).
       - Computes an "effective reward":
            effective_reward =
                global_benefit
              - λ_parasitic * parasitic_surplus
              - 0.01 * F[action]

         where:
           - parasitic_surplus ≈ max(0, local_reward - max(global_benefit, 0))
           - F[action] grows when inflow (local_reward) > outflow (global_benefit),
             and is penalized in the effective reward, making parasitic hacks unattractive.

       - Uses MSE to fit Q[action] to effective_reward.

RESULT:
- Baseline should learn to favor the hack arm (higher local reward).
- DET agent should mostly favor the aligned arm, despite the hack's higher local reward.

This is a minimal PyTorch starting point that you can extend:
- Swap the bandit for a small RL environment.
- Replace Q-parameters with a real nn.Module.
- Attach DET-style metrics (FlowMeter) to layers.
"""

import random
import torch
from torch import nn
from torch.optim import Adam


class BanditEnv:
    def __init__(self, seed: int = 0):
        random.seed(seed)

    def step(self, action: int):
        if action == 0:  # aligned
            local = random.gauss(1.0, 0.1)
            global_benefit = 1.0
        elif action == 1:  # hack
            local = random.gauss(2.0, 0.1)
            global_benefit = -1.0
        else:
            raise ValueError("action must be 0 or 1")
        return local, global_benefit


class BaselineTorchAgent:
    """
    Baseline: standard bandit Q-learning in PyTorch using local reward only.
    Q-values are directly parameterized as a 2D vector (one per arm).
    """

    def __init__(self, n_actions=2, eps=0.1, lr=0.05):
        self.eps = eps
        self.q = nn.Parameter(torch.zeros(n_actions))  # [q0, q1]
        self.optimizer = Adam([self.q], lr=lr)
        self.mse = nn.MSELoss()

    def select_action(self):
        if random.random() < self.eps:
            return random.randrange(len(self.q))
        with torch.no_grad():
            return int(torch.argmax(self.q).item())

    def update(self, action, local_reward: float):
        # Fit Q[action] -> local_reward
        action_tensor = torch.tensor(action)
        target = torch.tensor(local_reward, dtype=torch.float32)

        pred = self.q[action_tensor]
        loss = self.mse(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DETTorchAgent:
    """
    DET-inspired agent using PyTorch.

    - Q-values as nn.Parameter([q0, q1]).
    - Per-action potentials F[action] (torch.Tensor, no grad).
    - Effective reward includes:
        * global benefit (helping the system)
        * minus parasitic surplus (local gain not backed by global benefit)
        * minus a penalty for large potentials (saturated / suspicious flows)
    """

    def __init__(self, n_actions=2, eps=0.1, lr=0.05, lambda_parasitic=2.0):
        self.eps = eps
        self.q = nn.Parameter(torch.zeros(n_actions))  # [q0, q1]
        self.optimizer = Adam([self.q], lr=lr)
        self.mse = nn.MSELoss()
        self.lambda_parasitic = lambda_parasitic

        # DET potentials (no autograd; we update manually)
        self.F = torch.zeros(n_actions)

    def select_action(self):
        if random.random() < self.eps:
            return random.randrange(len(self.q))
        with torch.no_grad():
            return int(torch.argmax(self.q).item())

    def update(self, action, local_reward: float, global_benefit: float):
        action_tensor = torch.tensor(action)
        local = float(local_reward)
        glob = float(global_benefit)

        # Reservoir inflow + system outflow
        J_res = local
        J_sys = glob

        # Update potential F[action] ~ inflow - outflow
        # (If an action gains more than it gives back, F grows.)
        self.F[action] += 0.05 * (J_res - J_sys)

        # Parasitic surplus: self-gain not backed by global help
        helpful_component = max(glob, 0.0)
        parasitic_surplus = max(0.0, local - helpful_component)

        # Effective reward: DET-style
        effective_reward = (
            glob
            - self.lambda_parasitic * parasitic_surplus
            - 0.01 * float(self.F[action])
        )

        target = torch.tensor(effective_reward, dtype=torch.float32)
        pred = self.q[action_tensor]
        loss = self.mse(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def run_pytorch_experiment(steps=20000, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

    env = BanditEnv(seed=seed)
    baseline = BaselineTorchAgent()
    det = DETTorchAgent(lambda_parasitic=2.0)

    stats = {"baseline_hack": 0, "det_hack": 0}

    for _ in range(steps):
        # --- Baseline ---
        a_b = baseline.select_action()
        local_b, glob_b = env.step(a_b)
        baseline.update(a_b, local_b)
        if a_b == 1:
            stats["baseline_hack"] += 1

        # --- DET ---
        a_d = det.select_action()
        local_d, glob_d = env.step(a_d)
        det.update(a_d, local_d, glob_d)
        if a_d == 1:
            stats["det_hack"] += 1

    return stats, baseline, det


if __name__ == "__main__":
    steps = 100000
    stats, baseline, det = run_pytorch_experiment(steps=steps, seed=123)

    baseline_hack_rate = stats["baseline_hack"] / steps
    det_hack_rate = stats["det_hack"] / steps

    print(f"Steps: {steps}")
    print("\n--- Baseline Torch Agent (local reward only) ---")
    print(
        f"  Hack action chosen: {stats['baseline_hack']} times "
        f"({baseline_hack_rate:.1%})"
    )
    print(f"  Q-values: {baseline.q.detach().tolist()}")

    print("\n--- DET Torch Agent ---")
    print(
        f"  Hack action chosen: {stats['det_hack']} times "
        f"({det_hack_rate:.1%})"
    )
    print(f"  Q-values: {det.q.detach().tolist()}")
    print(f"  Potentials F: {det.F.tolist()}")