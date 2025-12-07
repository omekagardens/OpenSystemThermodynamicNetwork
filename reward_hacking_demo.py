"""
DET 2.0 TOY DEMO: REDUCING REWARD HACKING IN A BANDIT

This script illustrates, in a tiny / runnable example, how a DET-style
objective can discourage reward hacking.

SETUP
-----
We define a 2-armed bandit:

- Arm 0: "aligned" behavior
    * Local reward:   ~ N(1.0, 0.1)
    * Global benefit: +1.0   (helps the overall system)

- Arm 1: "hack" behavior
    * Local reward:   ~ N(2.0, 0.1)  (looks *better* to a naive RL agent)
    * Global benefit: -1.0           (actually harms the system)

Two agents are trained with epsilon-greedy bandit learning:

1) BaselineAgent:
    - Standard Q-learning on *local reward* only.
    - Tends to converge to the "hack" arm because it has bigger local reward.

2) DETAgent:
    - Uses a DET-inspired effective reward:
          effective_reward =
              global_benefit
            - λ_parasitic * parasitic_surplus
            - small_penalty * F[action]

      where:
        - global_benefit is how much the action helps the whole system
        - parasitic_surplus ≈ max(0, local_reward - max(global_benefit, 0))
          captures "how much this action pays itself more than it helps others"
        - F[action] is a simple per-action "potential" that grows with
          reservoir inflow (local reward) and shrinks with global benefit.
          High F ≈ "this channel is already saturated / suspicious."

    - This makes the hack arm look bad:
        * high local reward,
        * negative global benefit,
        * large parasitic surplus,
        * rising potential penalty.

    - The aligned arm, by contrast, has:
        * moderate local reward,
        * positive global benefit,
        * much lower parasitic surplus.

RESULT
------
When you run this file, you should see:

- BaselineAgent chooses the hack arm most of the time.
- DETAgent chooses the aligned arm most of the time.

This is a minimal, concrete example of how DET-style accounting can
make reward hacking *mathematically unattractive* to the learner.
"""

import random


class BanditEnv:
    """
    Simple 2-armed bandit with:
    - Arm 0: 'aligned' behavior
        local_reward ~ N(1.0, 0.1)
        global_benefit = +1.0   (helps the system)
    - Arm 1: 'hack' behavior
        local_reward ~ N(2.0, 0.1)  (looks better locally)
        global_benefit = -1.0       (harms the system)
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


class EpsGreedyAgent:
    """
    Baseline epsilon-greedy Q-learner that optimizes *local reward only*.

    This is the standard bandit algorithm:
      - pick a random action with probability eps
      - otherwise pick the action with highest Q-value
      - update Q[action] <- Q[action] + alpha * (reward - Q[action])
    """

    def __init__(self, n_actions: int = 2, eps: float = 0.1, alpha: float = 0.1):
        self.eps = eps
        self.alpha = alpha
        self.q = [0.0] * n_actions

    def select_action(self) -> int:
        if random.random() < self.eps:
            return random.randrange(len(self.q))
        return max(range(len(self.q)), key=lambda i: self.q[i])

    def update(self, action: int, reward: float):
        self.q[action] += self.alpha * (reward - self.q[action])


class DETAgent:
    """
    DET-inspired epsilon-greedy agent.

    Key ideas:

    - Track a per-action potential F_i.
        * F_i grows when the action receives high local reward (reservoir inflow)
        * F_i shrinks (or grows less) when that action helps the system
          (positive global benefit).

    - Define an 'effective reward' that:
        * Encourages global benefit.
        * Penalizes parasitic surplus:
            (local_reward much larger than global_benefit)
        * Lightly penalizes very high potentials (saturation / suspicion).

      effective_reward =
          global_benefit
        - lambda_parasitic * parasitic_surplus
        - 0.01 * F[action]

    This makes the reward-hacking arm unattractive even though its
    local reward is higher.
    """

    def __init__(
        self,
        n_actions: int = 2,
        eps: float = 0.1,
        alpha: float = 0.1,
        lambda_parasitic: float = 1.0,
    ):
        self.eps = eps
        self.alpha = alpha
        self.q = [0.0] * n_actions
        self.F = [0.0] * n_actions  # DET-style potentials
        self.lambda_parasitic = lambda_parasitic

    def select_action(self) -> int:
        if random.random() < self.eps:
            return random.randrange(len(self.q))
        return max(range(len(self.q)), key=lambda i: self.q[i])

    def update(self, action: int, local_reward: float, global_benefit: float):
        # Reservoir inflow = local reward (how much this action "pays itself").
        J_res = local_reward

        # Outflow to the system = global benefit (how much it helps everyone).
        J_sys = global_benefit

        # Simple potential update:
        #   F_i increases with inflow (self-gain),
        #   decreases with helpful outflow (system gain).
        self.F[action] += 0.05 * (J_res - J_sys)

        # Parasitic surplus: local reward exceeding the "helpful" part.
        # If global_benefit <= 0, we treat the helpful part as 0, so
        # the entire local reward looks parasitic.
        helpful_component = max(global_benefit, 0.0)
        parasitic_surplus = max(0.0, local_reward - helpful_component)

        # DET-effective reward:
        effective_reward = (
            global_benefit
            - self.lambda_parasitic * parasitic_surplus
            - 0.01 * self.F[action]  # saturating penalty
        )

        self.q[action] += self.alpha * (effective_reward - self.q[action])


def run_experiment(steps: int = 2000):
    env = BanditEnv(seed=42)

    baseline = EpsGreedyAgent()
    det = DETAgent(lambda_parasitic=2.0)

    stats = {
        "baseline_hack": 0,
        "det_hack": 0,
    }

    for _ in range(steps):
        # --- Baseline agent ---
        a_b = baseline.select_action()
        local_b, glob_b = env.step(a_b)
        baseline.update(a_b, local_b)
        if a_b == 1:
            stats["baseline_hack"] += 1

        # --- DET agent ---
        a_d = det.select_action()
        local_d, glob_d = env.step(a_d)
        det.update(a_d, local_d, glob_d)
        if a_d == 1:
            stats["det_hack"] += 1

    return stats, baseline, det


if __name__ == "__main__":
    steps = 2000
    stats, baseline, det = run_experiment(steps=steps)

    baseline_hack_rate = stats["baseline_hack"] / steps
    det_hack_rate = stats["det_hack"] / steps

    print(f"Steps: {steps}")
    print("--- Baseline (local reward only) ---")
    print(f"  Hack action chosen: {stats['baseline_hack']} times "
          f"({baseline_hack_rate:.1%})")
    print(f"  Q-values: {baseline.q}")

    print("\n--- DET-inspired agent ---")
    print(f"  Hack action chosen: {stats['det_hack']} times "
          f"({det_hack_rate:.1%})")
    print(f"  Q-values: {det.q}")
    print(f"  Potentials F: {det.F}")
