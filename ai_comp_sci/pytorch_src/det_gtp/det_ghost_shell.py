import networkx as nx
import numpy as np
import re
import operator
import random
from sklearn.linear_model import Ridge

# --- CONFIGURATION ---

# Physics
DIFFUSION_RATE      = 0.15    # Flow speed
DECAY_RATE          = 0.05    # Energy loss per tick
FATIGUE_RATE        = 0.15    # How fast a topic gets "boring"
RECOVERY_RATE       = 0.02    # How fast interest recovers
ACTIVATION_THRESH   = 1.0     # Minimum pressure to be "conscious"

# Sleep / Optimization
HISTORY_BUFFER_SIZE = 500     # How many ticks to remember before sleeping
MIN_TICKS_TO_SLEEP  = 50      # Minimum data points needed for regression
PRUNE_THRESHOLD     = 0.01    # If a synapse is weaker than this, kill it
RIDGE_ALPHA         = 1.0     # Regularization strength (prevents explosion)

class ResurrectionBrain:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_pressure = {}      # Current Voltage
        self.node_fatigue = {}       # Current Adaptation (Boredom)
        self.edge_conductance = {}   # Synaptic Weight
        
        # Short-term memory buffer for the "Resurrection" algorithm
        # Stores snapshots of brain activity: list of dicts {node: pressure}
        self.history_buffer = [] 
        
        self.tick_count = 0

    # --- 1. THE PHYSICS ENGINE (Wake State) ---

    def add_concept(self, name):
        if name not in self.node_pressure:
            self.graph.add_node(name)
            self.node_pressure[name] = 0.0
            self.node_fatigue[name] = 0.0

    def get_effective_pressure(self, node):
        """
        Pressure dampened by Fatigue. 
        This is what determines flow and conscious readout.
        """
        raw = self.node_pressure.get(node, 0.0)
        fatigue = self.node_fatigue.get(node, 0.0)
        # Denominator: As fatigue goes up, effective pressure crashes
        return raw / (1.0 + fatigue * 5.0)

    def tick(self, steps=1):
        """
        Advance time. 
        1. Flows pressure based on gradients.
        2. Increases fatigue on active nodes.
        3. Records state for the Sleep cycle.
        """
        for _ in range(steps):
            self.tick_count += 1
            
            # --- A. Diffusion (Flux) ---
            # Calculate flows based on EFFECTIVE pressure (so tired nodes don't push hard)
            deltas = {n: 0.0 for n in self.graph.nodes}
            
            for u, v in self.graph.edges:
                key = tuple(sorted((u, v)))
                sigma = self.edge_conductance.get(key, 0.1)
                
                eff_u = self.get_effective_pressure(u)
                eff_v = self.get_effective_pressure(v)
                
                # Flow from High to Low
                flow = (eff_u - eff_v) * DIFFUSION_RATE * sigma
                
                deltas[u] -= flow
                deltas[v] += flow

            # --- B. Update State ---
            current_activity = {}
            
            for n in self.graph.nodes:
                # Apply flow
                self.node_pressure[n] += deltas[n]
                
                # Apply Decay (Thermodynamics)
                self.node_pressure[n] *= (1.0 - DECAY_RATE)
                
                # Apply Fatigue (Biology)
                # If node is highly active, it gets tired
                if self.node_pressure[n] > ACTIVATION_THRESH:
                    self.node_fatigue[n] += FATIGUE_RATE
                
                # Recovery (Rest)
                self.node_fatigue[n] = max(0.0, self.node_fatigue[n] * (1.0 - RECOVERY_RATE))
                
                # Record for history if significant
                if abs(self.node_pressure[n]) > 0.01:
                    current_activity[n] = self.node_pressure[n]

            # --- C. Buffer for Sleep (Episodic RAM) ---
            # We record the raw activation state to learn correlations later
            if self.tick_count % 2 == 0: # Sample every other tick to save RAM
                self.history_buffer.append(current_activity)
                if len(self.history_buffer) > HISTORY_BUFFER_SIZE:
                    self.history_buffer.pop(0)

    # --- 2. THE RESURRECTION ENGINE (Sleep State) ---

    def sleep(self):
        """
        Consolidates memory using Ridge Regression.
        Calculates the matrix 'R' that best reconstructs the recent history.
        Rewrites the graph edges based on this mathematical truth.
        """
        if len(self.history_buffer) < MIN_TICKS_TO_SLEEP:
            return "Insomnia: Not enough experiences to sleep yet."

        print(f"\n[SLEEPING] Analyzing {len(self.history_buffer)} moments of history...")

        # 1. Prepare Data Matrix H (Time x Nodes)
        nodes = list(self.graph.nodes)
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)
        num_samples = len(self.history_buffer)

        # Build sparse-to-dense matrix
        H = np.zeros((num_samples, num_nodes))
        for t, snapshot in enumerate(self.history_buffer):
            for n, p in snapshot.items():
                if n in node_to_idx:
                    H[t, node_to_idx[n]] = p

        # 2. The Math: Ridge Regression
        # We want to find Matrix W such that H * W ≈ H
        # Meaning: Can I predict the state of Node B using the state of Node A?
        print("[SLEEPING] Dreaming (Solving Ridge Regression)...")
        
        # Fit H to H. The coef_ matrix is our new synaptic map.
        # fit_intercept=False because 0 input should mean 0 output (no ghost energy)
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        ridge.fit(H, H)
        
        # coef_ is shape (n_targets, n_features), which corresponds to (n_nodes, n_nodes)
        # Weight from j to i is stored in coef_[i, j]
        new_weights = ridge.coef_

        # 3. Rewire the Brain
        print("[SLEEPING] Rewiring synapses based on learned truths...")
        
        edges_added = 0
        edges_pruned = 0
        
        # We only care about the upper triangle (undirected graph)
        # or we average i->j and j->i for symmetry
        max_w = np.max(np.abs(new_weights)) if np.max(np.abs(new_weights)) > 0 else 1.0

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                u, v = nodes[i], nodes[j]
                
                # Average the bidirectional influence
                w_ij = new_weights[i, j]
                w_ji = new_weights[j, i]
                
                # Normalize strength relative to the strongest connection found
                avg_weight = (w_ij + w_ji) / 2.0
                normalized_weight = abs(avg_weight) # Magnitude only for conductance
                
                # Thresholding (Pruning)
                if normalized_weight < PRUNE_THRESHOLD:
                    if self.graph.has_edge(u, v):
                        self.graph.remove_edge(u, v)
                        del self.edge_conductance[tuple(sorted((u, v)))]
                        edges_pruned += 1
                else:
                    # Update or Create Edge
                    if not self.graph.has_edge(u, v):
                        self.graph.add_edge(u, v)
                        edges_added += 1
                    
                    self.edge_conductance[tuple(sorted((u, v)))] = normalized_weight * 5.0 # Scale up for physics

        # 4. Wake Up Fresh
        self.history_buffer = [] # Clear episodic RAM
        return f"Sleep Complete. Pruned {edges_pruned} weak links. Formed {edges_added} strong links."

    # --- 3. INPUT / OUTPUT ---

    def observe(self, text, strength=10.0):
        """
        Injects pressure. 
        Simulates 'Semantic Splashing' by activating immediate neighbors 
        to ensure the Ridge Regression has correlations to find.
        """
        words = [w.lower() for w in re.findall(r"\w+", text)]
        
        # 1. Direct Activation
        for w in words:
            self.add_concept(w)
            # Input overrides fatigue temporarily (Salience)
            self.node_pressure[w] += strength
            
        # 2. Simulated "Splash" (Temporary Associative Wiring)
        # In a real system, this would be an embedding vector injection.
        # Here, we wire the n-grams temporarily so the physics engine flows between them.
        # The Sleep cycle will solidify these if they repeat, or prune them if they don't.
        for i in range(len(words) - 1):
            u, v = words[i], words[i+1]
            if not self.graph.has_edge(u, v):
                self.graph.add_edge(u, v)
                # Give it a temporary "Fragile" connection
                self.edge_conductance[tuple(sorted((u, v)))] = 0.5

    def read_thoughts(self, top_k=5):
        candidates = [
            (n, self.get_effective_pressure(n)) 
            for n in self.graph.nodes 
            if self.get_effective_pressure(n) > ACTIVATION_THRESH
        ]
        # Sort by pressure
        candidates.sort(key=operator.itemgetter(1), reverse=True)
        
        if not candidates:
            return "..."
        
        return ", ".join([f"{n.upper()}({p:.1f})" for n, p in candidates[:top_k]])

# --- INTERACTIVE DEMO ---

if __name__ == "__main__":
    brain = ResurrectionBrain()
    
    print(">>> DET RESURRECTION ENGINE ONLINE <<<")
    print("Commands: /sleep, /stats, or just type text.")
    print("Notice how repeating a topic eventually makes the brain 'bored' (Fatigue).")
    print("Trigger /sleep to run the Ridge Regression optimization.\n")

    try:
        while True:
            user_in = input("Input: ").strip()
            
            if not user_in:
                brain.tick(5)
                continue

            if user_in == "/sleep":
                print(brain.sleep())
                continue
                
            if user_in == "/stats":
                print(f"Nodes: {brain.graph.number_of_nodes()}")
                print(f"Edges: {brain.graph.number_of_edges()}")
                print(f"History Samples: {len(brain.history_buffer)}")
                continue

            # 1. Input
            brain.observe(user_in)
            
            # 2. Process (The Thinking Loop)
            # Run a few ticks to let pressure spread and fatigue accumulate
            brain.tick(5)
            
            # 3. Readout
            thoughts = brain.read_thoughts()
            print(f"Brain: {thoughts}")
            
    except KeyboardInterrupt:
        print("\nSaving state... (just kidding, it's in RAM)")
