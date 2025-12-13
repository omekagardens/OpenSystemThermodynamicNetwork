import matplotlib
matplotlib.use('Agg') # Headless Mode

import torch
import numpy as np
import networkx as nx
import random

from det20_model import DETSystem 

# --- Random Config ---
# Set seed to None for randomness, or an integer for reproducibility
RANDOM_SEED = None 
if RANDOM_SEED:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

# --- Generator ---
def generate_random_plant(num_nodes=15):
    """Generates an organic tree structure."""
    # Barabasi-Albert for scale-free growth
    G = nx.barabasi_albert_graph(num_nodes, 1)
    
    # Simple physics layout to find top/bottom
    pos = nx.spring_layout(G, center=(0,0), iterations=50)
    y_vals = {i: pos[i][1] for i in range(num_nodes)}
    
    root_node = min(y_vals, key=y_vals.get)
    leaf_node = max(y_vals, key=y_vals.get)
    
    adj = torch.zeros(num_nodes, num_nodes)
    for u, v in G.edges():
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        
    return adj, root_node, leaf_node

# --- Setup ---
N = 20
adj, root_idx, leaf_idx = generate_random_plant(N)

BITE_TICK = random.randint(15, 35)
BITE_FORCE = random.uniform(3.5, 5.5)

print(f"--- CHAOS PARAMETERS ---")
print(f"Topology: Organic Tree ({N} Nodes)")
print(f"Root: {root_idx} | Leaf: {leaf_idx}")
print(f"Bite: Tick {BITE_TICK} | Force: {BITE_FORCE:.2f}")
print("-" * 30)

# Initialize
plant = DETSystem(num_nodes=N, phi_res=3.0, dt=0.05, noise_level=0.01)
plant.adj_energy = adj
plant.adj_info = adj # Shared topology for single plant
plant.a[root_idx] = 1.0

print(f"{'TICK':<6} | {'EVENT':<10} | {'ROOT(F)':<10} | {'LEAF(F)':<10} | {'SIGNAL':<8} | {'LOCKED':<8}")
print("-" * 70)

# --- Loop ---
for frame in range(80):
    bite_occurred = False
    
    # Chaos Bite
    if frame == BITE_TICK:
        plant.inject_trauma(leaf_idx, BITE_FORCE)
        bite_occurred = True

    plant.step(frame)
    stats = plant.telemetry
    
    event_msg = "!!! BITE !!!" if bite_occurred else ""
    
    # Log around the event
    if frame >= BITE_TICK - 5 and frame < BITE_TICK + 40:
         root_F = plant.F[root_idx].item()
         leaf_F = plant.F[leaf_idx].item()
         print(f"{frame:<6} | {event_msg:<10} | {root_F:<10.4f} | {leaf_F:<10.4f} | {stats['max_signal']:<8.4f} | {int(stats['nodes_locked']):<8}")