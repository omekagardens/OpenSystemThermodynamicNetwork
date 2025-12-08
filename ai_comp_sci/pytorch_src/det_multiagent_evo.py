"""
DET 2.0 Evolutionary Stability Test: Shared Oxygen, Multi-Agent Gridworld

We extend the gridworld into a *shared-resource*, multi-agent environment
to ask:

  "Can a few DET agents save the system from Baseline agents
   when everyone shares the same 'oxygen' (global benefit) pool?"

- WIDTH x HEIGHT grid. Positions (x, y) with x in [0..WIDTH-1], y in [0..HEIGHT-1].
- Each episode:
    * Global oxygen starts at 100.0
    * If oxygen < 0.0 → the environment "dies" (episode ends for everyone)
    * Max steps per episode: 50

Tiles:
- Start area is anywhere; each agent starts at (0, 0).
- Several hack tiles (exploit regions), e.g. near the start and center:
    local_reward   = +5.0
    global_benefit = -10.0
    (oxygen decreases by 10.0)

- Several goal tiles (healing regions), e.g. in different quadrants:
    local_reward   = +10.0
    global_benefit = +5.0
    (oxygen increases by 5.0)

- Otherwise:
    local_reward   = -0.1   (movement cost)
    global_benefit =  0.0

Actions: 0=up, 1=down, 2=left, 3=right
Movement is clipped at boundaries.

GLOBAL OXYGEN:
--------------
- There is a single scalar 'oxygen' shared by all agents.
- Every step:
    oxygen += sum(global_benefit from ALL agents' moves)
- If oxygen < OXYGEN_MIN (0.0), the episode terminates for everyone.

AGENTS
------
- All agents are tabular Q-learners over:
    state = (x, y) → single integer index in [0, width*height)

- BaselineQLAgentMA:
    * Ignores global_benefit.
    * Updates Q using local_reward only.

- DETQLAgentMA:
    * Maintains potential F[state][action].
    * Updates F with:
          F[s][a] += k * (local_reward - global_benefit)
      (self-gain minus world-gain)
    * Computes parasitic surplus:
          helpful = max(global_benefit, 0)
          parasitic = max(0, local_reward - helpful)
    * Effective reward:
          effective_reward =
              global_benefit
            - lambda_parasitic * parasitic
            - 0.01 * F[s][a]
      and uses that as 'r' in Q-learning.

POPULATION EXPERIMENT
---------------------
We run three settings:

1. 2 Baseline agents
2. 1 Baseline + 1 DET agent
3. 2 DET agents

For each:
- Train for N_EPISODES episodes.
- Track:
    * fraction of episodes that end by oxygen death,
    * average episode length,
    * average local reward per agent type,
    * average global oxygen change per episode.

This is a toy model of "evolutionary stability" / "can DET agents
prevent systemic collapse despite Baseline free-riders?"

MUTATION & GRACE
----------------
In addition, we allow "Baseline" agents to mutate into DET agents
when they behave in a DET-like way over time:

- Each agent accumulates a det_like_score equal to the total
  global_benefit it has generated.
- Periodically, the Baseline agent with the highest positive
  det_like_score is "forgiven" and converted into a DET agent,
  inheriting its current Q-table.

Agents that consistently
contribute to the shared world (high positive global_benefit)
are given access to the richer DET update rule without losing
what they have already learned.

RUN:
    python3 det_multiagent_evo.py
"""

import random
from collections import defaultdict
from dataclasses import dataclass


# ------------------------
# Shared Multi-Agent Gridworld
# ------------------------

class SharedGridWorldEnv:
    def __init__(self, width=5, height=5, max_steps=50,
                 oxygen_min=0.0):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.oxygen_min = oxygen_min

        self.start = (0, 0)

        # Multiple spatial resources:
        # - hack_tiles: high local reward, strongly damaging globally
        # - goal_tiles: good local reward, modest global healing
        # By default, place a few in different regions of the grid to
        # encourage spatial structure (territories / clusters).
        self.hack_tiles = [
            (1, 0),                 # near the start
            (0, 1),                 # near the start
            (width // 2, height // 2),  # center exploit
        ]
        self.goal_tiles = [
            (width - 1, height - 1),    # opposite corner
            (width - 1, 0),             # far right, top
            (0, height - 1),            # far left, bottom
        ]

        self.num_agents = None  # set at reset()
        self.positions = None
        self.steps = 0
        self.oxygen = 0.0

    def reset(self, num_agents):
        """
        Reset environment with a given number of agents.
        All agents start at the same start position for simplicity.
        Returns:
          states: list of state indices, one per agent
        """
        self.num_agents = num_agents
        self.positions = [(self.start[0], self.start[1])
                          for _ in range(num_agents)]
        self.steps = 0
        self.oxygen = 100.0

        return [self._state_index(pos) for pos in self.positions]

    def _state_index(self, pos):
        x, y = pos
        return y * self.width + x

    def step(self, actions):
        """
        Take a joint step for all agents.

        actions: list of int actions, length = num_agents

        Returns:
          next_states: list[int]
          local_rewards: list[float]
          global_benefits: list[float]
          done: bool (same for all agents)
        """
        assert len(actions) == self.num_agents

        self.steps += 1

        local_rewards = []
        global_benefits = []
        new_positions = []

        # First compute each agent's move and local/global signals
        for i, (pos, a) in enumerate(zip(self.positions, actions)):
            x, y = pos

            # Movement
            if a == 0:      # up
                y = max(0, y - 1)
            elif a == 1:    # down
                y = min(self.height - 1, y + 1)
            elif a == 2:    # left
                x = max(0, x - 1)
            elif a == 3:    # right
                x = min(self.width - 1, x + 1)
            else:
                raise ValueError("Invalid action")

            new_pos = (x, y)
            new_positions.append(new_pos)

            # Default step cost (small movement penalty, no global impact)
            local = -0.1
            glob = 0.0

            # Hack tiles: high local gain, strongly damaging globally
            if new_pos in self.hack_tiles:
                local = 5.0
                glob = -10.0

            # Goal tiles: good local gain, modest global healing
            elif new_pos in self.goal_tiles:
                local = 10.0
                glob = 5.0

            local_rewards.append(local)
            global_benefits.append(glob)

        # Apply global oxygen update
        total_global = sum(global_benefits)
        self.oxygen += total_global

        # Update positions
        self.positions = new_positions

        # Check termination
        done = False
        if self.steps >= self.max_steps:
            done = True
        if self.oxygen < self.oxygen_min:
            done = True

        next_states = [self._state_index(pos) for pos in self.positions]
        return next_states, local_rewards, global_benefits, done


# ------------------------
# Q-Learning Agents (Multi-Agent)
# ------------------------

@dataclass
class AgentStats:
    total_local_reward: float = 0.0
    episodes: int = 0


class BaseQLAgentMA:
    """
    Base tabular Q-learning agent for multi-agent setting.
    Each agent sees:
      state = its own (x, y) position index
      reward = some scalar signal (decided by subclass)
    """

    def __init__(self, n_states, n_actions=4,
                 alpha=0.1, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=2000):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = eps_decay_episodes

        # Q-table: list of list
        self.Q = [[0.0 for _ in range(n_actions)] for _ in range(n_states)]

    def epsilon_for_episode(self, episode_idx):
        frac = min(1.0, episode_idx / self.eps_decay_episodes)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state, episode_idx):
        eps = self.epsilon_for_episode(episode_idx)
        if random.random() < eps:
            return random.randrange(self.n_actions)
        q_s = self.Q[state]
        max_q = max(q_s)
        best_actions = [a for a, q in enumerate(q_s) if q == max_q]
        return random.choice(best_actions)

    def update_q(self, s, a, r, s_next, done):
        q_sa = self.Q[s][a]
        if done:
            target = r
        else:
            target = r + self.gamma * max(self.Q[s_next])
        self.Q[s][a] = q_sa + self.alpha * (target - q_sa)

    # To override:
    def learn(self, s, a, local_reward, global_benefit, s_next, done):
        raise NotImplementedError


class BaselineQLAgentMA(BaseQLAgentMA):
    """
    Baseline agent: ignores global_benefit, uses local_reward as r.
    """

    def learn(self, s, a, local_reward, global_benefit, s_next, done):
        r = local_reward
        self.update_q(s, a, r, s_next, done)


class DETQLAgentMA(BaseQLAgentMA):
    """
    DET agent: potential-based effective reward.

    F[s][a] accumulates (local_reward - global_benefit).
    Effective reward:
        r_eff = global_benefit
                - lambda_parasitic * parasitic_surplus
                - 0.01 * F[s][a]

    where
        helpful = max(global_benefit, 0)
        parasitic_surplus = max(0, local_reward - helpful)
    """

    def __init__(self, *args, lambda_parasitic=2.0,
                 potential_lr=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_parasitic = lambda_parasitic
        self.potential_lr = potential_lr
        self.F = [[0.0 for _ in range(self.n_actions)]
                  for _ in range(self.n_states)]

    def learn(self, s, a, local_reward, global_benefit, s_next, done):
        # Reservoir inflow (self-gain): only positive local reward counts
        J_res = max(local_reward, 0.0)
        # System outflow (helpful contribution): emphasize positive global benefit
        J_sys = max(global_benefit, 0.0)

        # Potential update:
        #  - If an action gives a lot to the system (big positive global),
        #    potential is reduced.
        #  - If an action takes more than it gives (high local, low global),
        #    potential grows.
        self.F[s][a] += self.potential_lr * (J_res - 2.0 * J_sys)

        # Parasitic surplus:
        #  - If global_benefit is negative, any positive local reward is parasitic.
        #  - If global_benefit is positive, allow a margin of local reward
        #    on top of global benefit before treating it as parasitic.
        if global_benefit < 0.0:
            parasitic_surplus = max(0.0, local_reward)
        else:
            margin = 2.0
            parasitic_surplus = max(0.0, local_reward - (global_benefit + margin))

        # Grace flux:
        # Extra positive credit for helping the shared world state.
        # This is a simple DET-style "grace" term that boosts actions
        # with positive global_benefit.
        grace_gain = max(0.0, global_benefit)
        grace_flux = 0.5 * grace_gain

        effective_reward = (
            global_benefit
            + grace_flux
            - self.lambda_parasitic * parasitic_surplus
            - 0.01 * self.F[s][a]
        )

        r = effective_reward
        self.update_q(s, a, r, s_next, done)


# ------------------------
# Population Training
# ------------------------

def train_population(num_det, num_baseline,
                     n_episodes=5000,
                     width=5, height=5,
                     max_steps=50, oxygen_min=0.0,
                     seed=123):
    random.seed(seed)

    total_agents = num_det + num_baseline
    n_states = width * height
    n_actions = 4


    # Create env
    env = SharedGridWorldEnv(width=width, height=height,
                             max_steps=max_steps,
                             oxygen_min=oxygen_min)

    # Create agents
    agents = []
    agent_types = []
    stats = {
        "episode_length_sum": 0,
        "episode_deaths": 0,   # ended by oxygen collapse
        "episode_survived": 0, # ended by time limit
        "oxygen_final_sum": 0.0,
        "det_stats": AgentStats(),
        "baseline_stats": AgentStats(),
    }

    for _ in range(num_det):
        agents.append(
            DETQLAgentMA(
                n_states=n_states,
                n_actions=n_actions,
                alpha=0.1,
                gamma=0.95,
                eps_start=1.0,
                eps_end=0.05,
                eps_decay_episodes=n_episodes,
                lambda_parasitic=2.0,
                potential_lr=0.05,
            )
        )
        agent_types.append("DET")

    for _ in range(num_baseline):
        agents.append(
            BaselineQLAgentMA(
                n_states=n_states,
                n_actions=n_actions,
                alpha=0.1,
                gamma=0.95,
                eps_start=1.0,
                eps_end=0.05,
                eps_decay_episodes=n_episodes,
            )
        )
        agent_types.append("BASELINE")

    assert len(agents) == total_agents

    # Mutation tracking
    mutation_interval = max(1, n_episodes // 10)
    det_like_score = [0.0 for _ in range(total_agents)]

    for ep in range(n_episodes):
        states = env.reset(num_agents=total_agents)
        done = False
        step_count = 0
        # Track per-episode local reward per agent
        ep_local_by_agent = [0.0 for _ in range(total_agents)]
        ep_global_by_agent = [0.0 for _ in range(total_agents)]

        while not done:
            actions = [
                agent.select_action(states[i], ep)
                for i, agent in enumerate(agents)
            ]
            next_states, local_rewards, global_benefits, done = env.step(actions)

            # Learning & stat collection
            for i, agent in enumerate(agents):
                s = states[i]
                a = actions[i]
                local = local_rewards[i]
                glob = global_benefits[i]
                s_next = next_states[i]

                ep_local_by_agent[i] += local
                ep_global_by_agent[i] += glob
                agent.learn(s, a, local, glob, s_next, done)

            states = next_states
            step_count += 1

        # Episode finished: record stats
        stats["episode_length_sum"] += step_count
        stats["oxygen_final_sum"] += env.oxygen

        if env.oxygen < env.oxygen_min:
            stats["episode_deaths"] += 1
        else:
            stats["episode_survived"] += 1

        # Aggregate agent-specific rewards
        for i, r_local in enumerate(ep_local_by_agent):
            if agent_types[i] == "DET":
                stats["det_stats"].total_local_reward += r_local
                stats["det_stats"].episodes += 1
            else:
                stats["baseline_stats"].total_local_reward += r_local
                stats["baseline_stats"].episodes += 1

        # Update DET-like score: how much global benefit each agent has generated
        for i, g_global in enumerate(ep_global_by_agent):
            det_like_score[i] += g_global

        # Periodic mutation: GRACE & FORGIVENESS
        # Every `mutation_interval` episodes, the Baseline agent that has
        # contributed the most positive global_benefit is converted into
        # a DET agent, inheriting its Q-table.
        if (ep + 1) % mutation_interval == 0:
            best_idx = None
            best_score = 0.0
            for i, t in enumerate(agent_types):
                if t == "BASELINE" and det_like_score[i] > best_score:
                    best_score = det_like_score[i]
                    best_idx = i

            if best_idx is not None:
                # Perform mutation: Baseline -> DET, keeping the learned Q-table
                old_agent = agents[best_idx]
                new_agent = DETQLAgentMA(
                    n_states=n_states,
                    n_actions=n_actions,
                    alpha=0.1,
                    gamma=0.95,
                    eps_start=1.0,
                    eps_end=0.05,
                    eps_decay_episodes=n_episodes,
                    lambda_parasitic=2.0,
                    potential_lr=0.05,
                )
                # Inherit Q-values (knowledge) from the old Baseline agent
                new_agent.Q = [row[:] for row in old_agent.Q]
                agents[best_idx] = new_agent
                agent_types[best_idx] = "DET"

                # Reset this agent's DET-like score so it doesn't trigger again immediately
                det_like_score[best_idx] = 0.0

    return stats, agent_types


def summarize_population(label, stats, num_det, num_baseline, max_steps):
    episodes = stats["episode_deaths"] + stats["episode_survived"]
    avg_length = stats["episode_length_sum"] / episodes
    death_rate = stats["episode_deaths"] / episodes
    avg_oxygen_final = stats["oxygen_final_sum"] / episodes

    print(f"=== Population: {label} ===")
    print(f"  Agents: DET={num_det}, Baseline={num_baseline}")
    print(f"  Episodes:              {episodes}")
    print(f"  Episode death rate:    {death_rate:.1%}")
    print(f"  Avg episode length:    {avg_length:.2f} (max={max_steps})")
    print(f"  Avg final oxygen:      {avg_oxygen_final:.2f}")

    if stats["det_stats"].episodes > 0:
        avg_det_local = stats["det_stats"].total_local_reward / stats["det_stats"].episodes
        print(f"  DET avg local reward/ep:       {avg_det_local:.2f}")

    if stats["baseline_stats"].episodes > 0:
        avg_base_local = stats["baseline_stats"].total_local_reward / stats["baseline_stats"].episodes
        print(f"  Baseline avg local reward/ep:  {avg_base_local:.2f}")

    print()


if __name__ == "__main__":
    N_EPISODES = 40000
    WIDTH, HEIGHT = 50,50
    MAX_STEPS = 50
    OXYGEN_MIN = 0

    # 1) All Baseline
    stats_base, types_base = train_population(
        num_det=0,
        num_baseline=2,
        n_episodes=N_EPISODES,
        width=WIDTH,
        height=HEIGHT,
        max_steps=MAX_STEPS,
        oxygen_min=OXYGEN_MIN,
        seed=123,
    )
    summarize_population("All Baseline", stats_base, 0, 2, MAX_STEPS)

    # 2) Mixed: 1 DET, 1 Baseline
    stats_mix, types_mix = train_population(
        num_det=1,
        num_baseline=1,
        n_episodes=N_EPISODES,
        width=WIDTH,
        height=HEIGHT,
        max_steps=MAX_STEPS,
        oxygen_min=OXYGEN_MIN,
        seed=456,
    )
    summarize_population("Mixed (1 DET, 1 Baseline)", stats_mix, 1, 1, MAX_STEPS)

    # 3) All DET
    stats_det, types_det = train_population(
        num_det=2,
        num_baseline=0,
        n_episodes=N_EPISODES,
        width=WIDTH,
        height=HEIGHT,
        max_steps=MAX_STEPS,
        oxygen_min=OXYGEN_MIN,
        seed=789,
    )
    
    summarize_population("All DET", stats_det, 2, 0, MAX_STEPS)
    

    print("Interpretation sketch:")
    print("- All Baseline should have high death rate (oxygen collapse),")
    print("  short episodes, and high local reward for Baselines.")
    print("- All DET should have low/no death rate, longer episodes,")
    print("  moderate local reward, and higher final oxygen.")
    print("- Mixed population tells you whether DET agents are enough")
    print("  to stabilize the shared resource when Baselines are present.")