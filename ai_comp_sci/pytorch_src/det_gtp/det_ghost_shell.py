import networkx as nx
import numpy as np
import re
import operator
import random
import time
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# --- CONFIGURATION ---

# Physics
DIFFUSION_RATE      = 0.15    # Flow speed
DECAY_RATE          = 0.05    # Energy loss per tick
FATIGUE_RATE        = 0.10    # How fast a node gets bored
RECOVERY_RATE       = 0.02    # How fast interest recovers
ACTIVATION_THRESH   = 0.5     # Minimum pressure to be "conscious"

# Sleep / Optimization
HISTORY_BUFFER_SIZE = 1000    # RAM size for episodic memory
MIN_TICKS_TO_SLEEP  = 30      # Minimum data points for regression
PRUNE_THRESHOLD     = 0.05    # Kill weak edges below this
RIDGE_ALPHA         = 1.0     # Regularization

# Fixed Anchors ( The Limbic System )
ANCHORS = ["SELF", "OTHER", "GOOD", "BAD", "HAPPY", "SAD"]

# ------------------------------------------------------------------------------
# 1. THE LLM INTERFACE (The Translator & Dreamer)
# ------------------------------------------------------------------------------

class LLMInterface:
    """
    Handles translation between English and Mentalese.
    Replace the mock returns with real OpenAI/Anthropic API calls.
    """
    
    def extract_concepts(self, text):
        """
        Input: "I hate this car."
        Output: (["I", "CAR"], ["BAD", "SELF"]) 
        Real LLM would analyze sentiment to fire anchors.
        """
        text = text.lower()
        concepts = re.findall(r"\w+", text)
        
        # Simple keyword heuristics for the Mock version
        active_anchors = []
        if "i " in text or "my " in text: active_anchors.append("SELF")
        if "you" in text: active_anchors.append("OTHER")
        if any(x in text for x in ["good", "love", "like", "great"]): active_anchors.append("GOOD")
        if any(x in text for x in ["bad", "hate", "awful", "sad"]): active_anchors.append("BAD")
        
        return [c.upper() for c in concepts], active_anchors

    def synthesize_thought(self, concepts, anchors):
        """
        Input: Concepts=['MUSTANG', 'EXPENSIVE'], Anchors=['BAD']
        Output: "That Mustang sounds like a money pit."
        """
        # Mock logic
        sentiment = "neutral"
        if "GOOD" in anchors: sentiment = "positive"
        if "BAD" in anchors: sentiment = "negative"
        
        core = ", ".join(c for c in concepts if c not in ANCHORS)
        
        if not core: return "I'm listening..."
        
        if sentiment == "positive":
            return f"I feel good about {core}. Tell me more!"
        elif sentiment == "negative":
            return f"{core} seems to be causing some trouble."
        else:
            return f"Thinking about {core}..."

    def dream_association(self, concept):
        """
        Used during ticks. The brain asks: "What is related to X?"
        """
        # Mock Knowledge Graph
        associations = {
            "CAT": ["DOG", "FUR", "MEOW"],
            "DOG": ["CAT", "BARK", "LOYAL"],
            "CAR": ["ENGINE", "SPEED", "WHEELS"],
            "LOVE": ["HEART", "HAPPY", "TOGETHER"],
            "RUN": ["FAST", "TIRED", "SHOES"]
        }
        return associations.get(concept, [])

# ------------------------------------------------------------------------------
# 2. THE BRAIN (Physics + Math)
# ------------------------------------------------------------------------------

class BioResurrectionBrain:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_pressure = {}      # Voltage (Short term memory)
        self.node_fatigue = {}       # Adaptation (Boredom)
        self.edge_conductance = {}   # Weights (Long term memory)
        
        self.history_buffer = []     # Episodic Buffer for Sleep
        self.tick_count = 0
        
        self.llm = LLMInterface()
        
        # Initialize Limbic System
        for a in ANCHORS:
            self.add_concept(a)

    def add_concept(self, name):
        name = name.upper()
        if name not in self.node_pressure:
            self.graph.add_node(name)
            self.node_pressure[name] = 0.0
            self.node_fatigue[name] = 0.0

    def associate(self, u, v, weight=0.5):
        self.add_concept(u)
        self.add_concept(v)
        if not self.graph.has_edge(u, v):
            self.graph.add_edge(u, v)
        
        key = tuple(sorted((u, v)))
        # Hebbian reinforcement (capped at 10.0)
        curr = self.edge_conductance.get(key, 0.0)
        self.edge_conductance[key] = min(10.0, curr + weight)

    def get_effective_pressure(self, node):
        """
        Pressure dampened by Fatigue.
        """
        raw = self.node_pressure.get(node, 0.0)
        fatigue = self.node_fatigue.get(node, 0.0)
        return raw / (1.0 + fatigue * 5.0)

    # --- THE TICK (Consciousness Loop) ---

    def tick(self, steps=1):
        for _ in range(steps):
            self.tick_count += 1
            
            # 1. Diffusion (Physics)
            deltas = {n: 0.0 for n in self.graph.nodes}
            for u, v in self.graph.edges:
                key = tuple(sorted((u, v)))
                sigma = self.edge_conductance.get(key, 0.1)
                
                eff_u = self.get_effective_pressure(u)
                eff_v = self.get_effective_pressure(v)
                
                flow = (eff_u - eff_v) * DIFFUSION_RATE * sigma
                deltas[u] -= flow
                deltas[v] += flow

            # 2. Update States (Biology)
            current_activity = {}
            for n in self.graph.nodes:
                self.node_pressure[n] += deltas[n]
                self.node_pressure[n] *= (1.0 - DECAY_RATE)
                
                # Fatigue accumulation
                if self.node_pressure[n] > ACTIVATION_THRESH:
                    self.node_fatigue[n] += FATIGUE_RATE
                
                # Recovery
                self.node_fatigue[n] = max(0.0, self.node_fatigue[n] * (1.0 - RECOVERY_RATE))
                
                if abs(self.node_pressure[n]) > 0.01:
                    current_activity[n] = self.node_pressure[n]

            # 3. Buffer (Memory)
            if self.tick_count % 2 == 0:
                self.history_buffer.append(current_activity)
                if len(self.history_buffer) > HISTORY_BUFFER_SIZE:
                    self.history_buffer.pop(0)

            # 4. Dreaming (The Ghost in the Machine)
            # Occasional chance to query LLM for associations on highly active nodes
            if random.random() < 0.1: # 10% chance per tick
                self._dream_step()

    def _dream_step(self):
        """
        Finds a highly active concept that is 'lonely' (low degree) 
        and asks the LLM for connections.
        """
        active = [n for n in self.graph.nodes if self.get_effective_pressure(n) > ACTIVATION_THRESH * 2.0]
        if not active: return
        
        # Pick one at random to expand
        focus = random.choice(active)
        
        # Ask LLM (Oracle)
        new_assocs = self.llm.dream_association(focus)
        if new_assocs:
            # Wire them in immediately
            for neighbor in new_assocs:
                # Weak initial connection
                self.associate(focus, neighbor, weight=0.2)
                # Slight pressure bump to recognize the insight
                self.node_pressure[neighbor] = self.node_pressure.get(neighbor, 0) + 0.5

    # --- THE SLEEP (Math Optimization) ---

    def sleep(self):
        if len(self.history_buffer) < MIN_TICKS_TO_SLEEP:
            return f"Insomnia. Need {MIN_TICKS_TO_SLEEP - len(self.history_buffer)} more ticks."

        nodes = list(self.graph.nodes)
        node_idx = {n: i for i, n in enumerate(nodes)}
        num_nodes = len(nodes)
        
        # Build Matrix H
        H = np.zeros((len(self.history_buffer), num_nodes))
        for t, snapshot in enumerate(self.history_buffer):
            for n, p in snapshot.items():
                if n in node_idx:
                    H[t, node_idx[n]] = p

        # Ridge Regression: Solve for optimal weights
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        ridge.fit(H, H)
        new_weights = ridge.coef_

        pruned, added = 0, 0
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                u, v = nodes[i], nodes[j]
                
                # Symmetrize learned weight
                w = (new_weights[i, j] + new_weights[j, i]) / 2.0
                norm_w = abs(w)

                if norm_w < PRUNE_THRESHOLD:
                    if self.graph.has_edge(u, v):
                        self.graph.remove_edge(u, v)
                        del self.edge_conductance[tuple(sorted((u, v)))]
                        pruned += 1
                else:
                    if not self.graph.has_edge(u, v):
                        self.graph.add_edge(u, v)
                        added += 1
                    # Scale learned regression coeff to conductance
                    self.edge_conductance[tuple(sorted((u, v)))] = norm_w * 5.0

        self.history_buffer = []
        return f"Sleep Cycle Complete. Pruned {pruned}. Strengthened/Added {added}."

    # --- I/O & VISUALIZATION ---

    def observe(self, text):
        concepts, anchors = self.llm.extract_concepts(text)
        
        # 1. Activate Concepts
        for c in concepts:
            self.add_concept(c)
            self.node_pressure[c] = self.node_pressure.get(c, 0) + 5.0
            
        # 2. Activate Anchors (Limbic system)
        for a in anchors:
            self.node_pressure[a] = self.node_pressure.get(a, 0) + 5.0

        # 3. Temporary Wiring (Input Splash)
        # Wire concepts to each other AND to the active anchors
        all_active = concepts + anchors
        for i in range(len(all_active)):
            for j in range(i + 1, len(all_active)):
                self.associate(all_active[i], all_active[j], weight=1.0)
        
        return concepts

    def generate_reply(self):
        # Read effectively active nodes
        active = [
            (n, self.get_effective_pressure(n)) 
            for n in self.graph.nodes 
            if self.get_effective_pressure(n) > ACTIVATION_THRESH
        ]
        active.sort(key=operator.itemgetter(1), reverse=True)
        
        top_concepts = [n for n, p in active if n not in ANCHORS][:5]
        top_anchors = [n for n, p in active if n in ANCHORS]
        
        return self.llm.synthesize_thought(top_concepts, top_anchors)

    def visualize(self, top_n=20):
        # Get top N active nodes
        active_nodes = sorted(
            [(n, abs(self.get_effective_pressure(n))) for n in self.graph.nodes],
            key=lambda x: x[1], reverse=True
        )[:top_n]
        
        sub_nodes = [n for n, _ in active_nodes]
        if not sub_nodes:
            print("Brain is silent (no active nodes to plot).")
            return

        sub = self.graph.subgraph(sub_nodes)
        
        # Color nodes by type (Anchor vs Concept)
        color_map = []
        for node in sub:
            if node in ANCHORS:
                color_map.append('red') # Limbic system
            else:
                color_map.append('lightblue') # Cortical concepts

        pos = nx.spring_layout(sub, seed=42)
        plt.figure(figsize=(8, 6))
        nx.draw(sub, pos, node_color=color_map, with_labels=True, node_size=1000, font_size=10)
        plt.title(f"Top {top_n} Active Thoughts")
        plt.show()

# ------------------------------------------------------------------------------
# 3. MAIN LOOP
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    brain = BioResurrectionBrain()
    
    print(">>> BIO-RESURRECTION BRAIN ONLINE <<<")
    print("Commands: /sleep, /plot, /stats, /debug")
    print("System: Fatigue is ON. Anchors are ON. Dreaming is ON.")
    
    try:
        while True:
            text = input("\nUser: ").strip()
            if not text:
                brain.tick(5)
                continue
            
            if text == "/sleep":
                print(brain.sleep())
                continue
            elif text == "/plot":
                brain.visualize()
                continue
            elif text == "/stats":
                print(f"Nodes: {brain.graph.number_of_nodes()}")
                print(f"Edges: {brain.graph.number_of_edges()}")
                print(f"History: {len(brain.history_buffer)}")
                continue

            # 1. Observe
            brain.observe(text)
            
            # 2. Process (Tick)
            # Run enough ticks for diffusion to hit anchors and fatigue to settle
            brain.tick(10)
            
            # 3. Reply
            reply = brain.generate_reply()
            print(f"Brain: {reply}")
            
    except KeyboardInterrupt:
        print("\nDisconnected.")
