import networkx as nx
import numpy as np
import operator

# --- DET 2.0-ALIGNED CONFIGURATION (CHATBOT ENGINE) ---
DIFFUSION_RATE = 0.15      # How fast "pressure" diffuses along edges
DECAY_RATE = 0.05          # How fast pressure fades (entropy / internal dissipation)
STEPS_PER_THOUGHT = 30     # How long to "think" before speaking
ACTIVATION_THRESHOLD = 2.0 # Minimum pressure to be considered an activated concept

# Reservoir coupling (open system: prevents full decay to zero)
RESERVOIR_LEVEL = 10.0        # Baseline "rest" potential
RESERVOIR_COUPLING = 0.05     # Coupling strength to the reservoir

class DETGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_pressure = {}
        self.edge_conductance = {}
        self.vocabulary = set()

    def add_concept(self, name):
        name = name.lower()
        if name not in self.vocabulary:
            self.graph.add_node(name)
            self.node_pressure[name] = 0.0
            self.vocabulary.add(name)

    def associate(self, n1, n2, weight=1.0):
        n1, n2 = n1.lower(), n2.lower()
        self.add_concept(n1)
        self.add_concept(n2)
        self.graph.add_edge(n1, n2)
        key = tuple(sorted((n1, n2)))
        self.edge_conductance[key] = weight

    def reset_mind(self):
        for n in self.node_pressure:
            self.node_pressure[n] = 0.0

    def think(self, steps=STEPS_PER_THOUGHT):
        # The Physics Engine
        for _ in range(steps):
            fluxes = {n: 0.0 for n in self.graph.nodes}
            
            for (u, v) in self.graph.edges:
                key = tuple(sorted((u, v)))
                cond = self.edge_conductance.get(key, 1.0)
                
                p_u = self.node_pressure[u]
                p_v = self.node_pressure[v]
                
                # Flow Equation
                flow = (p_u - p_v) * DIFFUSION_RATE * cond
                fluxes[u] -= flow
                fluxes[v] += flow

            # Apply Flux, Reservoir Injection & Decay (DET 2.0 style)
            for n in self.graph.nodes:
                # Flux from neighbors
                self.node_pressure[n] += fluxes[n]

                # Open-system reservoir coupling: inject toward RESERVOIR_LEVEL
                # Only inject when below reservoir level (no withdrawal here)
                delta_res = RESERVOIR_COUPLING * max(0.0, RESERVOIR_LEVEL - self.node_pressure[n])
                self.node_pressure[n] += delta_res

                # Internal dissipation
                self.node_pressure[n] *= (1.0 - DECAY_RATE)

    def chat(self, user_input):
        print(f"\nUser: '{user_input}'")
        
        # 1. Tokenize & Inject (The Prompt)
        words = user_input.lower().split()
        active_inputs = []

        # Simple "algebra mode" for 3 tokens: w0 - w1 + w2
        algebra_mode = (len(words) == 3)

        for i, w in enumerate(words):
            if w in self.vocabulary:
                if algebra_mode and i == 1:
                    # subtractive injection for the middle token
                    self.node_pressure[w] -= 100.0
                else:
                    # normal positive injection
                    self.node_pressure[w] += 100.0
                active_inputs.append(w)
        
        if not active_inputs:
            return "I don't understand those words yet."

        # 2. Process (The Thought)
        # Use fewer steps in algebra mode to keep the effect local
        if algebra_mode:
            self.think(steps=10)
        else:
            self.think()

        # 3. Retrieve Response (Emergent Activation)
        # Distance-aware scoring: prefer nodes that are close to multiple inputs
        results = {}
        for node, pressure in self.node_pressure.items():
            if node in active_inputs or pressure <= ACTIVATION_THRESHOLD:
                continue

            # Compute distances to all active inputs where a path exists
            dists = []
            for src in active_inputs:
                try:
                    if nx.has_path(self.graph, node, src):
                        dists.append(nx.shortest_path_length(self.graph, node, src))
                except nx.NetworkXNoPath:
                    continue

            if not dists:
                continue

            # Multi-source proximity: nodes central to several inputs get boosted
            proximity = sum(1.0 / (1.0 + d) for d in dists)
            score = pressure * proximity
            results[node] = score

        # Sort by distance-aware multi-source score
        sorted_thoughts = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
        
        # 4. Formulate Output
        if not sorted_thoughts:
            return "..."
        
        top_thought = sorted_thoughts[0][0]
        context = [t[0] for t in sorted_thoughts[1:4]]
        
        return f"Thinking of: {top_thought.upper()} (Context: {', '.join(context)})"

# --- BUILD THE BRAIN (KNOWLEDGE BASE) ---
bot = DETGraph()

# 1. Royalty Cluster
bot.associate("king", "queen", 0.9)
bot.associate("king", "crown", 0.9)
bot.associate("king", "prince", 0.7)
bot.associate("king", "palace", 0.8)
bot.associate("king", "rule", 0.8)
bot.associate("queen", "princess", 0.7)
bot.associate("crown", "gold", 0.6)

# 2. Family Cluster
bot.associate("king", "man", 0.8)
bot.associate("queen", "woman", 0.8)
bot.associate("prince", "boy", 0.8)
bot.associate("princess", "girl", 0.8)
bot.associate("man", "father", 0.7)
bot.associate("woman", "mother", 0.7)

# 3. Action/Need Cluster
bot.associate("eat", "food", 0.9)
bot.associate("eat", "hungry", 0.8)
bot.associate("drink", "water", 0.9)
bot.associate("drink", "thirsty", 0.8)
bot.associate("sleep", "bed", 0.9)
bot.associate("sleep", "tired", 0.8)

# 4. Conflict Cluster
bot.associate("fight", "sword", 0.9)
bot.associate("fight", "war", 0.9)
bot.associate("sword", "steel", 0.7)
bot.associate("king", "war", 0.6) # Kings go to war

# 5. Bridging Concepts (The Glue)
bot.associate("feast", "eat", 0.7)
bot.associate("feast", "king", 0.5) # Kings feast
bot.associate("feast", "wine", 0.6)
bot.associate("drink", "wine", 0.8)

# --- RUN THE CHATBOT ---
print(">>> DET-GPT ALPHA IS ONLINE <<<")
print("(Vocabulary: King, Queen, Eat, Drink, Fight, Sleep, Man, Woman...)")

# Test Cases
responses = []
responses.append(bot.chat("King")) # Expect: CROWN or QUEEN
bot.reset_mind()

responses.append(bot.chat("King Man Woman")) # Algebra: Expect QUEEN
bot.reset_mind()

responses.append(bot.chat("King Eat")) # Association: Expect FEAST or FOOD
bot.reset_mind()

responses.append(bot.chat("Fight")) # Expect: SWORD or WAR
bot.reset_mind()

for r in responses:
    print(f"Bot: {r}")