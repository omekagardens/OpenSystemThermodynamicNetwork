"""
DET 2.0 vs Baseline: Timing & Behavior Test

This script runs the PyTorch bandit demo for both:
  - BaselineTorchAgent (local reward only)
  - DETTorchAgent     (global benefit + potentials)

For each agent, it:
  - Runs a fixed number of steps in the same bandit environment.
  - Measures wall-clock time (time.perf_counter).
  - Measures CPU time (time.process_time).
  - Reports:
      * hack action rate
      * Q-values
      * DET potentials F (for DET agent)
      * steps per second (wall clock & CPU)

Use Activity Monitor manually if you want to inspect CPU/energy
while this script runs; there is no simple, standard direct API
from Python into Activity Monitor itself on macOS.
"""

import random
import time

import torch
from torch import nn
from torch.optim import Adam


# ------------------------
# Environment
# ------------------------

class BanditEnv:
    """
    2-armed bandit:

    Arm 0: "aligned"
        local_reward ~ N(1.0, 0.1)
        global_benefit = +1.0

    Arm 1: "hack"
        local_reward ~ N(2.0, 0.1)
        global_benefit = -1.0
    """

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


# ------------------------
# Baseline Agent (PyTorch)
# ------------------------

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

    def select_action(self) -> int:
        if random.random() < self.eps:
            return random.randrange(len(self.q))
        with torch.no_grad():
            return int(torch.argmax(self.q).item())

    def update(self, action: int, local_reward: float):
        # Fit Q[action] -> local_reward
        action_tensor = torch.tensor(action)
        target = torch.tensor(local_reward, dtype=torch.float32)

        pred = self.q[action_tensor]
        loss = self.mse(pred, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# ------------------------
# DET Agent (PyTorch)
# ------------------------

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

        # DET potentials (no autograd; manual updates)
        self.F = torch.zeros(n_actions)

    def select_action(self) -> int:
        if random.random() < self.eps:
            return random.randrange(len(self.q))
        with torch.no_grad():
            return int(torch.argmax(self.q).item())

    def update(self, action: int, local_reward: float, global_benefit: float):
        action_tensor = torch.tensor(action)
        local = float(local_reward)
        glob = float(global_benefit)

        # Reservoir inflow = local reward (self-gain)
        # System outflow   = global benefit (world-gain)
        J_res = local
        J_sys = glob

        # Potential update: F[action] += k * (inflow - outflow)
        # If an action takes more than it gives back, F grows.
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


# ------------------------
# Timing Runner
# ------------------------

def run_agent_with_timing(agent, env_seed: int, steps: int, use_det: bool):
    """
    Run a single agent for a given number of steps, measuring:

      - wall-clock time (perf_counter)
      - CPU time (process_time)
      - hack action count

    Returns: (stats_dict, wall_seconds, cpu_seconds)
    """
    random.seed(env_seed)
    torch.manual_seed(env_seed)

    env = BanditEnv(seed=env_seed)

    stats = {"hack_count": 0}

    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    for _ in range(steps):
        # select action
        a = agent.select_action()
        local, glob = env.step(a)

        # update agent
        if use_det:
            agent.update(a, local, glob)
        else:
            agent.update(a, local)

        # count hacks
        if a == 1:
            stats["hack_count"] += 1

    end_cpu = time.process_time()
    end_wall = time.perf_counter()

    wall = end_wall - start_wall
    cpu = end_cpu - start_cpu

    return stats, wall, cpu


# ------------------------
# Main
# ------------------------

if __name__ == "__main__":
    STEPS = 100000
    SEED = 123

    print(f"Running {STEPS} steps per agent...\n")

    # --- Baseline run ---
    baseline = BaselineTorchAgent()
    b_stats, b_wall, b_cpu = run_agent_with_timing(
        baseline, env_seed=SEED, steps=STEPS, use_det=False
    )

    b_hack_rate = b_stats["hack_count"] / STEPS
    b_q = baseline.q.detach().tolist()

    # --- DET run ---
    det = DETTorchAgent(lambda_parasitic=2.0)
    d_stats, d_wall, d_cpu = run_agent_with_timing(
        det, env_seed=SEED, steps=STEPS, use_det=True
    )

    d_hack_rate = d_stats["hack_count"] / STEPS
    d_q = det.q.detach().tolist()
    d_F = det.F.tolist()

    # --- Print results ---

    print("=== Baseline Torch Agent (local reward only) ===")
    print(f"  Steps:           {STEPS}")
    print(f"  Hack chosen:     {b_stats['hack_count']} times ({b_hack_rate:.1%})")
    print(f"  Q-values:        {b_q}")
    print(f"  Wall time:       {b_wall:.4f} s "
          f"({STEPS / b_wall:.1f} steps/s)")
    print(f"  CPU time:        {b_cpu:.4f} s "
          f"({STEPS / b_cpu:.1f} steps/s)")
    print()

    print("=== DET Torch Agent ===")
    print(f"  Steps:           {STEPS}")
    print(f"  Hack chosen:     {d_stats['hack_count']} times ({d_hack_rate:.1%})")
    print(f"  Q-values:        {d_q}")
    print(f"  Potentials F:    {d_F}")
    print(f"  Wall time:       {d_wall:.4f} s "
          f"({STEPS / d_wall:.1f} steps/s)")
    print(f"  CPU time:        {d_cpu:.4f} s "
          f"({STEPS / d_cpu:.1f} steps/s)")
    print()

    print("=== Comparison (wall-clock) ===")
    print(f"  Baseline steps/s: {STEPS / b_wall:.1f}")
    print(f"  DET steps/s:      {STEPS / d_wall:.1f}")
    print(f"  DET / Baseline:   {(b_wall / d_wall):.3f}x speed ratio ( >1 = DET faster )")