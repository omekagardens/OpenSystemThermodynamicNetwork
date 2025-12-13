import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

# Import the model from the separate file
from det20_model import DETSystem 

# --- Topology Generator (Unchanged) ---
def generate_plant_topology():
    num_nodes = 15
    adj = torch.zeros(num_nodes, num_nodes)
    pos = {} 
    
    # -- Positions --
    # Roots
    pos[0] = (0, -1.0); pos[1] = (-0.5, -1.5); pos[2] = (0.5, -1.5)
    # Stem
    for i in range(3, 9):
        pos[i] = (0, (i-3) * 0.8) 
    # Branch 1 (Left)
    pos[9] = (-0.8, 1.0); pos[10] = (-1.6, 1.2); pos[11] = (-2.2, 1.4)
    # Branch 2 (Right - Top)
    pos[12] = (0.8, 3.5); pos[13] = (1.6, 3.7); pos[14] = (2.2, 3.9)

    # -- Connections --
    connections = [
        (0, 1), (0, 2), # Roots
        (0, 3),         # Root -> Stem
        (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), # Stem
        (4, 9), (9, 10), (10, 11), # Branch 1
        (7, 12), (12, 13), (13, 14) # Branch 2
    ]
    
    for u, v in connections:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
        
    return num_nodes, adj, pos

# --- Initialize Simulation ---
N, adj_matrix, positions = generate_plant_topology()
plant = DETSystem(num_nodes=N, phi_res=3.0, dt=0.05)
plant.adj = adj_matrix

# Soil Access: Only Roots (0,1,2) have a=1
plant.a[0] = 1.0; plant.a[1] = 1.0; plant.a[2] = 1.0

# --- Console Logger Setup ---
print(f"{'TICK':<6} | {'EVENT':<10} | {'ROOT (F)':<10} | {'LEAF (F)':<10} | {'SYS ENERGY':<12} | {'COND (σ)':<10}")
print("-" * 75)

# --- Visualization ---
fig, ax = plt.subplots(figsize=(6, 8))
G = nx.from_numpy_array(plant.adj.numpy())

def update(frame):
    ax.clear()
    
    # Run Physics
    current_F = plant.step(frame)
    stats = plant.telemetry # Get the fresh data
    
    # --- CONSOLE LOGGING ---
    # Log every 5 ticks OR if something major happens (like a Bite)
    event_msg = ""
    if stats['is_bite']:
        event_msg = "!!! BITE !!!"
    
    if frame % 5 == 0 or stats['is_bite']:
        print(f"{frame:<6} | {event_msg:<10} | {stats['root_pressure']:<10.4f} | {stats['leaf_stress']:<10.4f} | {stats['total_energy']:<12.4f} | {stats['avg_conductivity']:<10.4f}")

    # --- GRAPHICS ---
    node_colors = [current_F[i] for i in range(N)]
    # Scale node size: Base size 100 + (Conductivity * 500)
    node_sizes = [plant.sigma[i].item() * 500 + 100 for i in range(N)]
    
    ax.axhline(y=-0.5, color='brown', linestyle='--', alpha=0.5, label="Soil Level")
    
    nx.draw_networkx_nodes(G, positions, node_size=node_sizes, 
                           node_color=node_colors, cmap=plt.cm.magma, 
                           vmin=0, vmax=3.0, ax=ax)
    nx.draw_networkx_edges(G, positions, edge_color='green', width=2, alpha=0.4, ax=ax)
    
    ax.set_title(f"DET 2.0 Plant Monitor - Tick: {frame}")
    
    if stats['is_bite']:
        ax.text(positions[14][0], positions[14][1]+0.3, "BITE!", color='red', fontsize=14, fontweight='bold')

ani = animation.FuncAnimation(fig, update, frames=100, interval=100, repeat=False)
plt.show()