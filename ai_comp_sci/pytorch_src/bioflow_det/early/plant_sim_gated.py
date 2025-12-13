import matplotlib
matplotlib.use('Agg') # Headless Mode

import torch
import numpy as np

from det20_model import DETSystem 

# --- Generator ---
def generate_plant_topology():
    num_nodes = 15
    adj_E = torch.zeros(num_nodes, num_nodes)
    adj_I = torch.zeros(num_nodes, num_nodes)
    
    # Stem & Branches
    connections = [
        (0, 1), (0, 2), (0, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
        (4, 9), (9, 10), (10, 11), (7, 12), (12, 13), (13, 14)
    ]
    for u, v in connections:
        adj_E[u, v] = 1.0; adj_E[v, u] = 1.0
        adj_I[u, v] = 1.0; adj_I[v, u] = 1.0
        
    return num_nodes, adj_E, adj_I

# --- Setup ---
N, adj_E, adj_I = generate_plant_topology()

plant = DETSystem(num_nodes=N, phi_res=3.0, dt=0.05, noise_level=0.01)
plant.adj_energy = adj_E
plant.adj_info = adj_I
plant.a[0] = 1.0; plant.a[1] = 1.0; plant.a[2] = 1.0

print("--- DET 2.0 SINGLE PLANT: PAUL REVERE DEMO ---")
print("Scenario: Bite at Tick 20")
print("-" * 70)
print(f"{'TICK':<6} | {'EVENT':<10} | {'ROOT(F)':<10} | {'LEAF(F)':<10} | {'SIGNAL':<8} | {'LOCKED':<8}")
print("-" * 70)

# --- Loop ---
for frame in range(80):
    bite_occurred = False
    
    if frame == 20:
        plant.inject_trauma(14, 4.0)
        bite_occurred = True

    plant.step(frame)
    stats = plant.telemetry
    
    event_msg = "!!! BITE !!!" if bite_occurred else ""
    
    if frame >= 18 and frame < 60:
        if frame % 2 == 0 or bite_occurred:
             # Calculate root pressure manually for display
             root_pressure = plant.F[0:3].mean().item()
             print(f"{frame:<6} | {event_msg:<10} | {root_pressure:<10.4f} | {plant.F[14]:<10.4f} | {stats['max_signal']:<8.4f} | {int(stats['nodes_locked']):<8}")