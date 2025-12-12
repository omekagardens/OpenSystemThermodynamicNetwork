import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json

# --- DET 9.0 CONFIGURATION (THE GRAPH ENGINE) ---
DIFFUSION_RATE = 0.10   # Flow speed along edges
DECAY_RATE = 0.05       # Entropy (stabilizes the system)
STEPS = 200
DATA_OUTPUT_FILE = "det_graph_data.json"

class DETGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_pressure = {}
        self.edge_conductance = {}
        self.history = []

    def add_node(self, name):
        self.graph.add_node(name)
        self.node_pressure[name] = 0.0

    def add_edge(self, n1, n2, weight=1.0):
        self.graph.add_edge(n1, n2)
        # Store conductance in a dictionary for fast lookup
        key = tuple(sorted((n1, n2)))
        self.edge_conductance[key] = weight

    def inject(self, name, amount):
        if name in self.node_pressure:
            self.node_pressure[name] += amount

    def step(self, step_count):
        # 1. Calculate Flux
        fluxes = {n: 0.0 for n in self.graph.nodes}
        
        for (u, v) in self.graph.edges:
            key = tuple(sorted((u, v)))
            cond = self.edge_conductance.get(key, 1.0)
            
            p_u = self.node_pressure[u]
            p_v = self.node_pressure[v]
            
            # Flow from High to Low
            flow = (p_u - p_v) * DIFFUSION_RATE * cond
            
            fluxes[u] -= flow
            fluxes[v] += flow

        # 2. Apply Flux & Decay
        for n in self.graph.nodes:
            self.node_pressure[n] += fluxes[n]
            self.node_pressure[n] *= (1.0 - DECAY_RATE)

        # 3. Inputs (The "Royal Algebra")
        # King - Man + Woman = ?
        self.inject("King", 10.0)
        self.inject("Woman", 10.0)
        self.inject("Man", -10.0)

        # 4. Logging
        snapshot = {
            "step": step_count,
            "pressures": self.node_pressure.copy()
        }
        self.history.append(snapshot)
        
        return self.node_pressure

    def save_data(self):
        print(f"\nSaving graph data to {DATA_OUTPUT_FILE}...")
        with open(DATA_OUTPUT_FILE, 'w') as f:
            json.dump(self.history, f, indent=4)
        print("Save Complete.")

# --- INITIALIZATION ---
det = DETGraph()
nodes = ["King", "Man", "Woman", "Queen"]
for n in nodes:
    det.add_node(n)

# Build the Semantic Square
# King is related to Man and Woman
det.add_edge("King", "Man", 0.8)
det.add_edge("King", "Woman", 0.5)

# Woman is related to Queen (Strong)
det.add_edge("Woman", "Queen", 0.8)

# Man is related to Queen (Weak/Negative relation structurally?)
# In this graph, we connect them to allow flow to interact
det.add_edge("Man", "Queen", 0.5)

# --- VISUALIZATION SETUP ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("DET 9.0: Graph Engine (Semantic Algebra)")

# Fixed Layout for consistency
pos = {
    "King":  [-1, 1],
    "Man":   [1, 1],
    "Woman": [-1, -1],
    "Queen": [1, -1]
}

def animate(i):
    if i == STEPS - 1:
        det.save_data()
    
    pressures = det.step(i)
    
    ax.clear()
    ax.set_title(f"DET Graph Engine | Step {i}")
    
    # Color Mapping
    # Red = Positive (Source/Target)
    # Blue = Negative (Sink)
    node_colors = []
    node_sizes = []
    
    for n in det.graph.nodes:
        p = pressures[n]
        # Normalize for visual color (assuming range -50 to +100)
        if p >= 0:
            # Positive: Fade to Red
            intensity = np.clip(p / 80.0, 0, 1)
            node_colors.append((1.0, 0.0, 0.0, intensity))
        else:
            # Negative: Fade to Blue
            intensity = np.clip(abs(p) / 20.0, 0, 1)
            node_colors.append((0.0, 0.0, 1.0, intensity))
            
        node_sizes.append(1000 + abs(p)*10)

    # Draw Graph
    nx.draw_networkx_nodes(det.graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors='black')
    nx.draw_networkx_labels(det.graph, pos, ax=ax, font_color='black', font_weight='bold')
    
    # Draw Edges with width based on conductance (static for now)
    nx.draw_networkx_edges(det.graph, pos, ax=ax, width=2, edge_color='gray')
    
    # Draw Edge Labels (Conductance)
    edge_labels = {k: f"{v:.1f}" for k, v in det.edge_conductance.items()}
    # nx.draw_networkx_edge_labels(det.graph, pos, edge_labels=edge_labels, ax=ax)

    # Annotate Pressure Values
    for n, (x, y) in pos.items():
        p = pressures[n]
        ax.text(x, y-0.15, f"{p:.1f}", fontsize=10, ha='center', color='black', fontweight='bold')

print("Simulating Graph Dynamics...")
ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=50, blit=False, repeat=False)
plt.show()