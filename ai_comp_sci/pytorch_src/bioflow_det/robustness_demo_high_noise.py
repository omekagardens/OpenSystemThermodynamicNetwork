from __future__ import annotations

import random
import statistics
from typing import List, Dict, Optional, Tuple

from det_core import DETConfig, DETSystem
from det_topology import DETTopology
from det_labs import DETLab

# Functions from research_demo_conditioning.py
def pick_distinct_nodes(rng: random.Random, lo: int, hi: int, k: int, avoid: set[int]) -> List[int]:
    pool = [i for i in range(lo, hi + 1) if i not in avoid]
    rng.shuffle(pool)
    return pool[:k]

def analyze_bite(
    history: List[Dict],
    bite_tick: int,
    nodes_per_plant: int,
    I_thresh: float = 0.05,
    window: int = 15,
) -> Dict:
    n = len(history)
    t_end = min(bite_tick + window, n - 1)

    prev_t = max(bite_tick - 1, 0)
    baseline_locked_B = float(history[prev_t].get("locked_B", 0.0))
    baseline_I = float(history[prev_t].get("I_bridge_B", 0.0))
    saturated_at_bite = baseline_locked_B >= float(nodes_per_plant - 1)

    cross_tick = None
    resp_tick = None
    peak_locked_B_delta = 0.0
    peak_I_amp = 0.0

    for t in range(bite_tick, n):
        I_amp = float(history[t].get("I_bridge_B", 0.0))
        locked_B = float(history[t].get("locked_B", 0.0))
        if t <= t_end:
            peak_I_amp = max(peak_I_amp, I_amp)

        if cross_tick is None and I_amp >= I_thresh:
            cross_tick = t

        if resp_tick is None and locked_B > baseline_locked_B + 0.5:
            resp_tick = t

        if t <= t_end:
            peak_locked_B_delta = max(peak_locked_B_delta, locked_B - baseline_locked_B)

        if t > t_end and cross_tick is not None and resp_tick is not None:
            break

    latency = None
    if cross_tick is not None and resp_tick is not None:
        latency = resp_tick - cross_tick

    return {
        "bite_tick": bite_tick,
        "baseline_locked_B": baseline_locked_B,
        "saturated_at_bite": saturated_at_bite,
        "bridge_cross_tick": cross_tick,
        "first_lock_B_tick": resp_tick,
        "latency_lock_minus_cross": latency,
        "peak_locked_B_delta_window": peak_locked_B_delta,
        "peak_I_bridge_B_window": peak_I_amp,
    }

def robustness_test(seed=42, steps=150, noise_level=0.1, bite_ticks=[30, 60, 90], log_freq=10):
    random.seed(seed)
    config = DETConfig(phi_res=3.0, noise_level=noise_level, threshold_stress=3.0)
    
    nodes_per_plant = 20
    adj_E, adj_I, roots, leaves = DETTopology.create_fungal_bridge(nodes_per_plant)
    meta = DETTopology.bridge_metadata(nodes_per_plant, roots, leaves)
    
    forest = DETSystem(num_nodes=40, config=config)
    forest.adj_energy = adj_E
    forest.adj_info = adj_I
    forest.a[roots[0]] = 1.0
    forest.a[roots[1]] = 1.0
    
    for i in range(40):
        forest.activate_node(i)
    
    rng = random.Random(seed)
    avoid = {roots[0]}
    bite_nodes = pick_distinct_nodes(rng, 0, nodes_per_plant - 1, k=len(bite_ticks), avoid=avoid)
    
    all_events = list(zip(bite_ticks, bite_nodes, [5.0] * len(bite_ticks), ["BITE"] * len(bite_ticks)))
    events_by_tick = {t: (node, force, label) for (t, node, force, label) in all_events}
    
    def bite_schedule(sim):
        if sim.tick in events_by_tick:
            node, force, label = events_by_tick[sim.tick]
            print(f">>> EVENT: {label} bite injected at Node {node} (Force {force})")
            DETLab.inject_chaos_bite(sim, node, force=force)
    
    forest.register_hook("pre_step", bite_schedule)
    
    history = DETLab.run_headless(forest, steps=steps, log_freq=log_freq, meta=meta)
    
    metrics = []
    for t, node, _, label in all_events:
        m = analyze_bite(history, t, nodes_per_plant, I_thresh=0.05, window=20)
        m["label"] = label
        m["leaf"] = node
        metrics.append(m)
    
    resp_ok = sum(1 for m in metrics if m['first_lock_B_tick'] is not None)
    cross_ok = sum(1 for m in metrics if m['bridge_cross_tick'] is not None)
    print(f"Cross OK: {cross_ok}/{len(metrics)} | Resp OK: {resp_ok}/{len(metrics)}")
    return metrics, float(resp_ok / len(metrics)) if len(metrics) > 0 else 0.0

if __name__ == "__main__":
    # Example run with high noise and 4 bites
    metrics, rate = robustness_test(seed=3, steps=150, noise_level=0.2, bite_ticks=[30, 60, 90, 120], log_freq=10)

    # Batch over 5 runs for average
    rates = []
    for s in range(5):
        _, r = robustness_test(seed=s, steps=150, noise_level=0.2, bite_ticks=[30, 60, 90, 120], log_freq=0)
        rates.append(r)

    mean_rate = statistics.mean(rates)
    print(f"Average response rate over 5 runs with noise 0.2: {mean_rate*100:.0f}%")
