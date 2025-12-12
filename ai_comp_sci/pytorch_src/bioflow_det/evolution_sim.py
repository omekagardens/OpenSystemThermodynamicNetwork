import matplotlib
matplotlib.use('Agg') # Headless

import torch
import numpy as np
import networkx as nx
import random

from det20_model_v4 import DETSystemV4 

# --- Setup ---
MAX_NODES = 50
sim = DETSystemV4(max_nodes=MAX_NODES, phi_res=8.0, dt=0.05, noise_level=0.01)

# --- GENESIS: Plant a single seed ---
# Node 0 is the First Root. It has high energy to start.
sim.activate_node(idx=0, energy=2.0, conductivity=1.0, soil_access=1.0)

print("--- DET 2.0 V4: EVOLUTIONARY DYNAMICS ---")
print("Starting with 1 Seed. Max Capacity: 50.")
print("Rules: F > 5.0 -> Mitosis | F <= 0 -> Death")
print("-" * 60)
print(f"{'TICK':<6} | {'ALIVE':<6} | {'ENERGY':<8} | {'BIRTHS':<6} | {'DEATHS':<6}")
print("-" * 60)

# --- Loop ---
for frame in range(150):
    
    # Run Physics + Evolution
    current_F = sim.step(frame)
    stats = sim.telemetry
    
    if frame % 5 == 0 or stats['births'] > 0 or stats['deaths'] > 0:
        print(f"{frame:<6} | {int(stats['alive_count']):<6} | {stats['total_energy']:<8.2f} | {stats['births']:<6} | {stats['deaths']:<6}")

    # --- TOPOLOGY SNAPSHOT (Optional Visualization Logic) ---
    # In a real GUI, we would rebuild the graph here based on sim.adj_energy
    if stats['births'] > 0:
        # Just creating a log entry for now
        pass