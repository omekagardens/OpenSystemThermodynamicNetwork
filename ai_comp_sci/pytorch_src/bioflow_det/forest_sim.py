import matplotlib
matplotlib.use('Agg') # Headless Mode

import torch
import numpy as np
import sys

# Import from our new model file
from det20_model import DETSystem 

# --- Ecosystem Generator ---
def generate_ecosystem():
    """
    Creates TWO plants (A and B).
    Nodes 0-14: Plant A (Left)
    Nodes 15-29: Plant B (Right)
    """
    total_nodes = 30
    adj_E = torch.zeros(total_nodes, total_nodes)
    adj_I = torch.zeros(total_nodes, total_nodes)
    
    def build_plant(offset_idx):
        connections = [
            (0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
            (4, 9), (9, 10), (10, 11), (7, 12), (12, 13), (13, 14)
        ]
        for u, v in connections:
            idx_u, idx_v = u+offset_idx, v+offset_idx
            adj_E[idx_u, idx_v] = 1.0; adj_E[idx_v, idx_u] = 1.0
            adj_I[idx_u, idx_v] = 1.0; adj_I[idx_v, idx_u] = 1.0

    build_plant(0)  # Plant A
    build_plant(15) # Plant B
    
    # --- FUNGAL BRIDGE ---
    # High conductivity (5.0) for Information only between roots
    adj_I[0, 15] = 5.0
    adj_I[15, 0] = 5.0
    
    return total_nodes, adj_E, adj_I

# --- Setup ---
N, adj_E, adj_I = generate_ecosystem()

# Initialize with Noise
forest = DETSystem(num_nodes=N, phi_res=3.0, dt=0.05, noise_level=0.01)
forest.adj_energy = adj_E
forest.adj_info = adj_I

# Soil Access
forest.a[0] = 1.0; forest.a[1] = 1.0; forest.a[2] = 1.0
forest.a[15] = 1.0; forest.a[16] = 1.0; forest.a[17] = 1.0

print("--- DET 2.0 ECOSYSTEM: HEADLESS MODE ---")
print(f"Nodes: {N} | Connection: Fungal Bridge (A <-> B)")
print("Scenario: Bite Plant A -> Watch Plant B Lock Down")
print("-" * 75)
print(f"{'TICK':<6} | {'EVENT':<10} | {'PLANT A (F)':<12} | {'BRIDGE SIGNAL':<14} | {'PLANT B LOCKED':<16}")
print("-" * 75)

# --- Loop ---
for frame in range(80):
    bite_occurred = False
    
    # Bite Logic
    if frame == 20:
        forest.inject_trauma(14, 4.0) # Bite Node 14
        bite_occurred = True

    # Physics
    forest.step(frame)
    stats = forest.telemetry
    
    # Logging
    event_msg = "!!! BITE !!!" if bite_occurred else ""
    should_log = (frame < 5) or (frame >= 18 and frame < 65) or (frame % 10 == 0)
    
    if should_log:
         print(f"{frame:<6} | {event_msg:<10} | {forest.F[14]:<12.4f} | {stats['bridge_signal']:<14.4f} | {int(stats['plant_B_locked']):<16}")