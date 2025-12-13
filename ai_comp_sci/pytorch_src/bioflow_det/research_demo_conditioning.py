# research_demo_conditioning.py
from __future__ import annotations

import random
import argparse
import statistics
from typing import List, Dict, Optional, Tuple

from det_core import DETConfig, DETSystem
from det_topology import DETTopology
from det_labs import DETLab


def pick_distinct_nodes(rng: random.Random, lo: int, hi: int, k: int, avoid: set[int]) -> List[int]:
    """Pick k distinct ints in [lo, hi] not in avoid."""
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
    """
    Compute per-bite metrics from recorded history (per-tick stats):
      - bridge_cross_tick_after_bite: first tick >= bite_tick where I_bridge_B >= I_thresh
      - first_lock_B_tick_after_bite: first tick >= bite_tick where locked_B > 0
      - latency: lock_tick - cross_tick (if both)
      - peak_locked_B_window: max locked_B in [bite_tick, bite_tick+window]
    """
    n = len(history)
    t_end = min(bite_tick + window, n - 1)

    # Baselines right BEFORE the bite (so we measure *changes caused by this bite*)
    prev_t = max(bite_tick - 1, 0)
    baseline_locked_B = float(history[prev_t].get("locked_B", 0.0))
    baseline_I = float(history[prev_t].get("I_bridge_B", 0.0))
    baseline_unlocked_B = max(0.0, float(nodes_per_plant) - baseline_locked_B)
    saturated_at_bite = baseline_locked_B >= float(nodes_per_plant - 1)

    cross_tick: Optional[int] = None
    resp_tick: Optional[int] = None
    peak_locked_B_delta = 0.0
    peak_I_amp = 0.0

    for t in range(bite_tick, n):
        I_amp = float(history[t].get("I_bridge_B", 0.0))
        locked_B = float(history[t].get("locked_B", 0.0))
        if t <= t_end:
            peak_I_amp = max(peak_I_amp, I_amp)

        # Crossing: first time bridge endpoint exceeds threshold AFTER this bite
        if cross_tick is None and I_amp >= I_thresh:
            cross_tick = t

        # Response: first time Plant B locking increases beyond its pre-bite baseline
        if resp_tick is None and locked_B > baseline_locked_B + 0.5:
            resp_tick = t

        if t <= t_end:
            peak_locked_B_delta = max(peak_locked_B_delta, locked_B - baseline_locked_B)

        # Early exit once we've passed the window and found both key times
        if t > t_end and cross_tick is not None and resp_tick is not None:
            break

    latency: Optional[int] = None
    if cross_tick is not None and resp_tick is not None:
        latency = resp_tick - cross_tick

    return {
        "bite_tick": bite_tick,
        "baseline_locked_B": baseline_locked_B,
        "baseline_unlocked_B": baseline_unlocked_B,
        "baseline_I_bridge_B": baseline_I,
        "saturated_at_bite": saturated_at_bite,
        "bridge_cross_tick": cross_tick,
        "first_lock_B_tick": resp_tick,
        "latency_lock_minus_cross": latency,
        "peak_locked_B_delta_window": peak_locked_B_delta,
        "peak_I_bridge_B_window": peak_I_amp,
    }


def run_once(seed: int | None = None, steps: int = 120, log_freq: int = 5, test_tick: int = 70) -> Dict:
    # --- Config ---
    config = DETConfig(
        phi_res=3.0,
        noise_level=0.01,
        threshold_stress=3.0
    )

    nodes_per_plant = 20
    total_nodes = nodes_per_plant * 2

    # --- Topology ---
    adj_E, adj_I, roots, leaves = DETTopology.create_fungal_bridge(nodes_per_plant=nodes_per_plant)
    plant_A_root, plant_B_root = roots
    plant_A_leaf, _plant_B_leaf = leaves

    meta = DETTopology.bridge_metadata(nodes_per_plant, roots, leaves)
    r1, r2 = meta["bridge_endpoints"]

    # --- Build system ---
    forest = DETSystem(num_nodes=total_nodes, config=config)
    forest.adj_energy = adj_E
    forest.adj_info = adj_I

    # Soil access only at roots (per-node gating vector)
    forest.a[:] = 0.0
    forest.a[plant_A_root] = 1.0
    forest.a[plant_B_root] = 1.0

    # Wake up all nodes
    for i in range(forest.N):
        forest.activate_node(i)

    # --- Conditioning protocol ---
    rng = random.Random(seed) if seed is not None else random.Random()

    # Pick 3 conditioning leaves on Plant A and 1 test leaf (generalization)
    avoid = {plant_A_root, plant_A_leaf}
    cond_leaves = pick_distinct_nodes(rng, 0, nodes_per_plant - 1, k=3, avoid=avoid)
    avoid.update(cond_leaves)
    test_leaf = pick_distinct_nodes(rng, 0, nodes_per_plant - 1, k=1, avoid=avoid)[0]

    # Conditioning + test schedule
    conditioning_ticks = [20, 35, 50]
    test_tick = test_tick
    all_events: List[Tuple[int, int, float, str]] = []  # (tick, node, force, label)

    for t, node in zip(conditioning_ticks, cond_leaves):
        all_events.append((t, node, 5.0, "COND"))
    all_events.append((test_tick, test_leaf, 5.0, "TEST"))

    events_by_tick = {t: (node, force, label) for (t, node, force, label) in all_events}

    def bite_schedule(sim: DETSystem):
        # Inject bites at scheduled ticks
        if sim.tick in events_by_tick:
            node, force, label = events_by_tick[sim.tick]
            print(f">>> EVENT: {label} bite injected at Node {node} (Force {force})")
            # Note: DETLab.inject_chaos_bite also prints a generic bite line.
            # Keeping only the labeled line helps avoid confusing duplicates.
            DETLab.inject_chaos_bite(sim, node, force=force)

    forest.register_hook("pre_step", bite_schedule)

    # --- Run ---
    if log_freq and log_freq > 0:
        print(f"Setup:")
        print(f"  Bridge endpoints: A_root={r1}, B_root={r2}")
        print(f"  Plant A root={plant_A_root}, Plant B root={plant_B_root}")
        print(f"  Conditioning leaves: {cond_leaves} at ticks {conditioning_ticks}")
        print(f"  Test leaf: {test_leaf} at tick {test_tick}")
        print()

    history = DETLab.run_headless(forest, steps=steps, log_freq=log_freq, meta=meta)

    # --- Post-run analysis (per bite) ---
    metrics = []
    for (t, node, _force, label) in all_events:
        m = analyze_bite(history, bite_tick=t, nodes_per_plant=nodes_per_plant, I_thresh=0.05, window=15)
        m["label"] = label
        m["leaf"] = node
        metrics.append(m)

    plant_b_indices = range(nodes_per_plant, total_nodes)
    final_locked_b = sum(1 for i in plant_b_indices if forest.g_lateral[i] < 0.5)

    return {
        "seed": seed,
        "setup": {
            "roots": roots,
            "bridge_endpoints": (r1, r2),
            "cond_leaves": cond_leaves,
            "test_leaf": test_leaf,
        },
        "metrics": metrics,
        "final_locked_b": final_locked_b,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1, help="number of runs to execute")
    ap.add_argument("--base-seed", type=int, default=0, help="base seed for reproducible batches")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--log-freq", type=int, default=5, help="0 disables periodic tick printing")
    ap.add_argument("--test-tick", type=int, default=70)
    args = ap.parse_args()

    results: List[Dict] = []
    for r in range(args.runs):
        seed = (args.base_seed + r) if args.runs > 1 else None
        # For batch runs, default to quiet unless user explicitly set a nonzero log-freq
        res = run_once(seed=seed, steps=args.steps, log_freq=args.log_freq, test_tick=args.test_tick)
        results.append(res)

        # If single run, print the per-run analysis table (keeps old behavior)
        if args.runs == 1:
            metrics = res["metrics"]
            print("\n--- CONDITIONING ANALYSIS (per-bite metrics) ---")
            print("tick | type | leaf | cross | lockBΔ | lat | peakΔB | peakI | sat")
            print("-" * 74)
            for m in metrics:
                t = m["bite_tick"]
                label = m["label"]
                node = m["leaf"]
                cross = m["bridge_cross_tick"]
                lockb = m["first_lock_B_tick"]
                lat = m["latency_lock_minus_cross"]
                peak_delta = m["peak_locked_B_delta_window"]
                peak_I = m["peak_I_bridge_B_window"]
                sat = m["saturated_at_bite"]

                cross_s = str(cross) if cross is not None else "-"
                lock_s = str(lockb) if lockb is not None else "-"
                lat_s = str(lat) if lat is not None else "-"
                sat_s = "Y" if sat else "N"
                print(f"{t:<4} | {label:<4} | {node:<4} | {cross_s:<5} | {lock_s:<5} | {lat_s:<3} | {peak_delta:<6.0f} | {peak_I:<5.2f} | {sat_s}")

            print("\n--- FINAL SNAPSHOT ---")
            print(f"Final Plant B locked nodes (gate closed): {res['final_locked_b']}")
            print("Note: 'cross' uses the bridge threshold; 'lockBΔ' is first tick Plant B locks beyond its pre-bite baseline; 'sat=Y' means Plant B was already near-fully locked at bite time.")
            return

    # Batch summary (runs > 1)
    # Focus on TEST event
    test_lat: List[int] = []
    test_peakI: List[float] = []
    test_peakDelta: List[float] = []
    test_cross_ok = 0
    test_resp_ok = 0
    sat_count = 0
    final_locked: List[int] = []

    for res in results:
        final_locked.append(int(res["final_locked_b"]))
        metrics = res["metrics"]
        test = next(m for m in metrics if m["label"] == "TEST")
        if test["saturated_at_bite"]:
            sat_count += 1
        if test["bridge_cross_tick"] is not None:
            test_cross_ok += 1
        if test["first_lock_B_tick"] is not None:
            test_resp_ok += 1
        if test["latency_lock_minus_cross"] is not None:
            test_lat.append(int(test["latency_lock_minus_cross"]))
        test_peakI.append(float(test["peak_I_bridge_B_window"]))
        test_peakDelta.append(float(test["peak_locked_B_delta_window"]))

    def mean(xs: List[float]) -> float:
        return float(statistics.mean(xs)) if xs else float("nan")

    def med(xs: List[float]) -> float:
        return float(statistics.median(xs)) if xs else float("nan")

    print(f"--- BATCH SUMMARY ---")
    print(f"Runs: {args.runs} | steps={args.steps} | test_tick={args.test_tick} | base_seed={args.base_seed}")
    print(f"TEST: cross_ok={test_cross_ok}/{args.runs} | resp_ok={test_resp_ok}/{args.runs} | saturated_at_test={sat_count}/{args.runs}")
    if test_lat:
        print(f"TEST latency (lockBΔ - cross): mean={mean([float(x) for x in test_lat]):.2f}  median={med([float(x) for x in test_lat]):.2f}  min={min(test_lat)}  max={max(test_lat)}")
    else:
        print("TEST latency (lockBΔ - cross): n/a (no responses)")
    print(f"TEST peakI (15t window): mean={mean(test_peakI):.2f}  median={med(test_peakI):.2f}")
    print(f"TEST peakΔB (15t window): mean={mean(test_peakDelta):.2f}  median={med(test_peakDelta):.2f}")
    print(f"Final locked_B: mean={mean([float(x) for x in final_locked]):.2f}  median={med([float(x) for x in final_locked]):.2f}  min={min(final_locked)}  max={max(final_locked)}")


if __name__ == "__main__":
    main()