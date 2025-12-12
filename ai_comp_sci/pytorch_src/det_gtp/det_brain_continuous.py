import networkx as nx
import operator
import re
import json

# --- DET 2.0-ALIGNED CONFIGURATION (CONTINUOUS BRAIN) ---

DIFFUSION_RATE      = 0.15   # How fast pressure diffuses along edges
DECAY_RATE          = 0.05   # Internal dissipation per substep
RESERVOIR_LEVEL     = 10.0   # Baseline rest potential
RESERVOIR_COUPLING  = 0.05   # Coupling strength to reservoir
HOMEOSTASIS_DECAY   = 0.0005 # Slow global decay to prevent long-term saturation

# Learning / plasticity
LEARNING_RATE       = 0.001  # How fast conductances strengthen when used
DECAY_SIGMA         = 0.0001 # Weight decay on conductances

# Co-activation growth
COACT_THRESHOLD     = 3.0    # Pressure threshold for co-activation
NEW_EDGE_SIGMA      = 0.05   # Initial conductance for new edges

# Readout
ACTIVATION_THRESHOLD = 2.0   # Minimum pressure to be considered "active" for readout

# Basic stopwords to avoid saturating on function words
STOPWORDS = {
    "the", "a", "an", "of", "and", "to", "in", "is", "it", "that", "i",
    "you", "he", "she", "we", "they", "this", "those", "these", "for",
    "on", "as", "with", "at", "by", "from", "be", "are", "was", "were"
}

# Structural DET terms that we route through but generally do not want as
# direct "answer" concepts in readout.
DET_STRUCTURAL_STOPWORDS = {"node", "nodes", "system", "systems"}

# Stability / clamping
MAX_PRESSURE        = 1000.0  # Hard cap on node pressure magnitude
MIN_SIGMA           = 1e-4    # Minimum edge conductance
MAX_SIGMA           = 10.0    # Maximum edge conductance
MAX_NEW_EDGES_STEP  = 20      # Limit on new edges formed per DET step
TARGET_PRESSURE_MAX = 100.0   # Desired upper bound for |pressure| before rescaling

# Optional: seed a tiny "royal" toy graph on first run.
# For DET-only experiments this should remain False.
ENABLE_ROYAL_SEED   = False


class ContinuousDETBrain:
    """
    A continuous, self-updating DET 2.0 brain.

    - Maintains persistent node pressures F_i.
    - Diffuses pressure along edges with dissipation + reservoir.
    - Adapts edge conductances based on flux (usage).
    - Creates new edges between frequently co-active concepts.
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.node_pressure = {}      # F_i
        self.edge_conductance = {}   # sigma_ij
        self.vocabulary = set()
        self.tick_index = 0        # total number of DET steps applied

    # --- Graph construction -------------------------------------------------

    def add_concept(self, name: str):
        name = name.lower()
        if name not in self.vocabulary:
            self.graph.add_node(name)
            self.node_pressure[name] = 0.0
            self.vocabulary.add(name)

    def associate(self, n1: str, n2: str, weight: float = 1.0):
        n1, n2 = n1.lower(), n2.lower()
        self.add_concept(n1)
        self.add_concept(n2)
        self.graph.add_edge(n1, n2)
        key = tuple(sorted((n1, n2)))
        self.edge_conductance[key] = weight

    # --- Core DET 2.0 dynamics ---------------------------------------------

    def _det_step(self):
        """
        One DET 2.0 update step with learning:
        - compute fluxes along edges
        - apply flux + reservoir + dissipation to node pressures
        - update edge conductances based on flux
        - create new edges from co-active nodes
        """
        if not self.graph.nodes:
            return

        # Track flux usage for learning
        edge_flux_accum = {key: 0.0 for key in self.edge_conductance}

        # 1. Compute fluxes
        fluxes = {n: 0.0 for n in self.graph.nodes}

        for (u, v) in self.graph.edges:
            key = tuple(sorted((u, v)))
            cond = self.edge_conductance.get(key, 1.0)

            p_u = self.node_pressure[u]
            p_v = self.node_pressure[v]

            # Flow from u->v (antisymmetric)
            flow = (p_u - p_v) * DIFFUSION_RATE * cond

            fluxes[u] -= flow
            fluxes[v] += flow

            edge_flux_accum[key] += abs(flow)

        # 2. Apply flux, reservoir injection, and decay to node pressures
        for n in self.graph.nodes:
            # neighbor flux
            self.node_pressure[n] += fluxes[n]

            # reservoir coupling: inject when below, drain when above
            delta_res = RESERVOIR_COUPLING * (RESERVOIR_LEVEL - self.node_pressure[n])
            self.node_pressure[n] += delta_res

            # dissipation
            self.node_pressure[n] *= (1.0 - DECAY_RATE)

            # slow homeostatic decay to prevent long-term saturation
            self.node_pressure[n] -= HOMEOSTASIS_DECAY * self.node_pressure[n]

            # clamp pressure for stability
            if self.node_pressure[n] > MAX_PRESSURE:
                self.node_pressure[n] = MAX_PRESSURE
            elif self.node_pressure[n] < -MAX_PRESSURE:
                self.node_pressure[n] = -MAX_PRESSURE

        # 2b. Automatic global renormalization to keep pressures in a sane range
        if self.node_pressure:
            current_max = max(abs(p) for p in self.node_pressure.values())
            if current_max > TARGET_PRESSURE_MAX:
                scale = TARGET_PRESSURE_MAX / current_max
                for n in self.node_pressure:
                    self.node_pressure[n] *= scale

        # 3. Update conductances based on usage (Hebbian-ish with decay)
        for key, usage in edge_flux_accum.items():
            # strengthen edges that carried flux
            self.edge_conductance[key] += LEARNING_RATE * usage
            # mild decay to prevent blow-up
            self.edge_conductance[key] *= (1.0 - DECAY_SIGMA)

            # clamp conductance within reasonable bounds
            if self.edge_conductance[key] < MIN_SIGMA:
                self.edge_conductance[key] = MIN_SIGMA
            elif self.edge_conductance[key] > MAX_SIGMA:
                self.edge_conductance[key] = MAX_SIGMA

        # 4. Grow new edges from co-activation
        active_nodes = [n for n, p in self.node_pressure.items() if p > COACT_THRESHOLD]

        new_edges_added = 0
        for i, a in enumerate(active_nodes):
            for b in active_nodes[i+1:]:
                if new_edges_added >= MAX_NEW_EDGES_STEP:
                    break
                if a == b:
                    continue
                if not self.graph.has_edge(a, b):
                    self.associate(a, b, weight=NEW_EDGE_SIGMA)
                    new_edges_added += 1
            if new_edges_added >= MAX_NEW_EDGES_STEP:
                break

    def tick(self, steps: int = 1):
        """
        Advance the brain by a number of DET steps.
        This is the 'time evolution' operator.
        """
        for _ in range(steps):
            self._det_step()
            self.tick_index += 1

    # --- Input: observing text / prompts -----------------------------------

    WORD_RE = re.compile(r"[a-zA-Z']+")

    def _tokenize(self, text: str):
        return [m.group(0).lower() for m in self.WORD_RE.finditer(text)]

    def observe_text(self, text: str, strength: float = 5.0):
        """
        Injects pressure based on a line of text.
        - Adds unseen words as new concepts.
        - Adds a modest positive spike to their pressure.
        """
        words = [w for w in self._tokenize(text) if w not in STOPWORDS]
        if not words:
            return []

        observed = []
        for w in words:
            self.add_concept(w)
            self.node_pressure[w] += strength
            observed.append(w)

        # Optional: encourage local association among simultaneously observed words
        # (a kind of immediate co-occurrence wiring)
        for i, a in enumerate(observed):
            for b in observed[i+1:]:
                if not self.graph.has_edge(a, b):
                    self.associate(a, b, weight=NEW_EDGE_SIGMA)

        return observed

    # --- Output: reading the brain's state ---------------------------------

    def read_thoughts(self, top_k: int = 5):
        """
        Read out the top-k active concepts by pressure.
        """
        candidates = [
            (n, p) for n, p in self.node_pressure.items()
            if p > ACTIVATION_THRESHOLD
        ]
        candidates.sort(key=operator.itemgetter(1), reverse=True)
        return candidates[:top_k]

    def read_contextual_thought_structured(self, query: str, top_k: int = 5):
        """
        Structured variant of contextual readout.
        - Observes the query (injects it).
        - Ticks the brain a bit.
        - Returns (top_concept, [context_concepts]) instead of a string.
        Scoring is based on delta-pressure caused by this query, so long-term
        hot clusters do not dominate every response.
        """
        # Snapshot baseline pressures before this query
        baseline = dict(self.node_pressure)

        observed = self.observe_text(query, strength=10.0)

        # Let the brain integrate this injection
        self.tick(steps=5)

        if not observed:
            return None, []

        # Distance & delta-pressure-based scoring relative to observed tokens
        results = {}
        for node, new_p in self.node_pressure.items():
            if node in observed:
                continue
            if node in DET_STRUCTURAL_STOPWORDS:
                # We still allow routing through these, but do not pick them
                # as the "headline" thought for conversational output.
                continue

            old_p = baseline.get(node, 0.0)
            delta = new_p - old_p
            # Only consider nodes that were actually excited by this query
            if delta <= 0 or delta <= ACTIVATION_THRESHOLD:
                continue

            # Compute distances to observed tokens
            dists = []
            for src in observed:
                try:
                    if nx.has_path(self.graph, node, src):
                        dists.append(nx.shortest_path_length(self.graph, node, src))
                except nx.NetworkXNoPath:
                    continue

            if not dists:
                continue

            proximity = sum(1.0 / (1.0 + d) for d in dists)
            # Degree-based penalty: very high-degree hubs get slightly down-weighted
            try:
                deg = self.graph.degree[node]
            except Exception:
                deg = 0
            score = delta * proximity / (1.0 + 0.1 * deg)
            results[node] = score

        if not results:
            return None, []

        sorted_thoughts = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
        top = sorted_thoughts[0][0]
        ctx = [t[0] for t in sorted_thoughts[1:1+top_k]]

        return top, ctx

    def read_contextual_thought(self, query: str, top_k: int = 5):
        """
        Like det_gpt_alpha.chat, but without reset and with learning.
        Returns a human-readable debug string.
        """
        top, ctx = self.read_contextual_thought_structured(query, top_k=top_k)
        if top is None:
            return "..."
        return f"Thinking of: {top.upper()} (Context: {', '.join(ctx)})"

    def compose_reply(self, query: str, top_k: int = 3):
        """
        Higher-level conversational wrapper:
        - Uses the DET field to select a concept + context.
        - Wraps that into a simple natural-language sentence.
        """
        top, ctx = self.read_contextual_thought_structured(query, top_k=top_k)
        if top is None:
            return "I'm not sure how to connect that yet, but I am listening."

        # Build a simple phrase from context words
        ctx_filtered = [w for w in ctx if w not in STOPWORDS]
        if ctx_filtered:
            ctx_phrase = ", ".join(ctx_filtered)
            return f"When you say '{query}', I find myself thinking about {top.lower()}, in relation to {ctx_phrase}."
        else:
            return f"When you say '{query}', I find myself thinking about {top.lower()}."

    # --- Offline / batch training from text sequences ---------------------

    def train_on_text_sequences(self, texts, strength: float = 5.0, steps_per_line: int = 5, epochs: int = 1):
        """
        Perform simple DET-style training on a list of text strings.
        For each line:
          - observe_text(line) to inject concepts
          - tick(steps_per_line) to let flows + learning update structure
        Repeats for the given number of epochs.
        """
        if not texts:
            return

        for epoch in range(epochs):
            for line in texts:
                self.observe_text(str(line), strength=strength)
                self.tick(steps=steps_per_line)

    def train_from_json(self, path: str, text_key: str = "sentences", strength: float = 5.0, steps_per_line: int = 5, epochs: int = 1):
        """
        Load a simple JSON corpus from disk and train on it.

        Supported formats:
          1) Object with a list of strings under `text_key` (default: "sentences"):
               { "sentences": ["King and queen sit in the palace.", "Love brings hope."] }

          2) Top-level JSON array of strings:
               ["Line one of training", "Another line"]

          3) Object with a list of objects, each having a `text` field:
               { "data": [ {"text": "foo"}, {"text": "bar"} ] }  (use text_key="data")
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        if isinstance(data, dict):
            # case 1: dict with list of strings under text_key
            value = data.get(text_key, [])
            if isinstance(value, list):
                if value and isinstance(value[0], str):
                    texts = value
                elif value and isinstance(value[0], dict) and "text" in value[0]:
                    # case 3: list of objects with "text" field
                    texts = [item.get("text", "") for item in value]
        elif isinstance(data, list):
            # case 2: direct list of strings
            texts = [str(x) for x in data]

        # Filter out empties
        texts = [t for t in texts if t]

        self.train_on_text_sequences(texts, strength=strength, steps_per_line=steps_per_line, epochs=epochs)
    # --- Global renormalization ------------------------------------------

    def rebalance_pressures(self, target_max: float = 50.0):
        """
        Rescale all node pressures so that the maximum absolute pressure
        equals `target_max`. This is useful if many nodes have saturated
        at the clamp ceiling.
        """
        if not self.node_pressure:
            return
        current_max = max(abs(p) for p in self.node_pressure.values())
        if current_max <= 0:
            return
        scale = target_max / current_max
        for n in self.node_pressure:
            self.node_pressure[n] *= scale

    # --- Introspection / stats --------------------------------------------

    def get_stats(self, top_k: int = 10):
        """
        Return a dictionary of basic brain stats for debugging/introspection.
        """
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        top_nodes = self.read_thoughts(top_k=top_k)
        return {
            "ticks": self.tick_index,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "top_nodes": top_nodes,
        }

    # --- Persistence: save / load brain state -----------------------------

    def to_dict(self):
        return {
            "tick_index": int(self.tick_index),
            "nodes": {
                n: float(self.node_pressure.get(n, 0.0))
                for n in self.graph.nodes
            },
            "edges": [
                {
                    "u": u,
                    "v": v,
                    "sigma": float(self.edge_conductance.get(tuple(sorted((u, v))), 1.0)),
                }
                for (u, v) in self.graph.edges
            ],
        }

    @classmethod
    def from_dict(cls, data):
        brain = cls()
        brain.tick_index = int(data.get("tick_index", 0))
        # restore nodes
        for n, p in data.get("nodes", {}).items():
            brain.add_concept(n)
            brain.node_pressure[n] = float(p)
        # restore edges
        for e in data.get("edges", []):
            u = e["u"]
            v = e["v"]
            sigma = float(e.get("sigma", 1.0))
            brain.associate(u, v, weight=sigma)
        return brain

    def save_state(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_state(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# --- Simple demo / REPL ----------------------------------------------------

if __name__ == "__main__":
    import os

    STATE_PATH = "det_brain_state.json"

    if os.path.exists(STATE_PATH):
        print(f"Loading brain state from {STATE_PATH}...")
        brain = ContinuousDETBrain.load_state(STATE_PATH)
    else:
        brain = ContinuousDETBrain()

    freshly_initialized = not os.path.exists(STATE_PATH)

    if freshly_initialized and ENABLE_ROYAL_SEED:
        # Optionally: seed with a few initial associations like in det_gpt_alpha
        brain.associate("king", "queen", 0.9)
        brain.associate("king", "crown", 0.9)
        brain.associate("king", "palace", 0.8)
        brain.associate("queen", "princess", 0.7)
        brain.associate("queen", "woman", 0.8)
        brain.associate("woman", "mother", 0.7)
        brain.associate("man", "father", 0.7)

    print(">>> Continuous DET Brain Online <<<")
    print("Type text; the brain will adapt over time. Ctrl+C to exit.")

    try:
        while True:
            line = input("\nUser: ")
            cmd = line.strip()

            # Introspection / control commands (do not inject text)
            if cmd.startswith("/"):
                if cmd in ("/stats", "/debug"):
                    stats = brain.get_stats(top_k=10)
                    print(f"[ticks={stats['ticks']}] nodes={stats['num_nodes']}, edges={stats['num_edges']}")
                    if stats["top_nodes"]:
                        print("Top nodes:")
                        for n, p in stats["top_nodes"]:
                            print(f"  - {n}: {p:.3f}")
                    brain.save_state(STATE_PATH)
                    continue
                elif cmd.startswith("/tick"):
                    parts = cmd.split()
                    try:
                        steps = int(parts[1]) if len(parts) > 1 else 1
                    except ValueError:
                        steps = 1
                    brain.tick(steps=steps)
                    thoughts = brain.read_thoughts()
                    print(f"[manual tick] advanced {steps} steps (tick_index={brain.tick_index})")
                    if thoughts:
                        print("Current thoughts:", thoughts)
                    brain.save_state(STATE_PATH)
                    continue
                elif cmd in ("/thoughts", "/top"):
                    thoughts = brain.read_thoughts()
                    if thoughts:
                        print("Active thoughts:")
                        for n, p in thoughts:
                            print(f"  - {n}: {p:.3f}")
                    else:
                        print("No active thoughts above threshold.")
                    brain.save_state(STATE_PATH)
                    continue
                elif cmd.startswith("/train"):
                    # Usage: /train path.json [epochs]
                    parts = cmd.split()
                    if len(parts) < 2:
                        print("Usage: /train path.json [epochs]")
                        continue
                    path = parts[1]
                    try:
                        epochs = int(parts[2]) if len(parts) > 2 else 1
                    except ValueError:
                        epochs = 1
                    if not os.path.exists(path):
                        print(f"File not found: {path}")
                        continue
                    print(f"Training from JSON: {path} (epochs={epochs})")
                    brain.train_from_json(path, text_key="sentences", strength=5.0, steps_per_line=3, epochs=epochs)
                    brain.save_state(STATE_PATH)
                    stats = brain.get_stats(top_k=10)
                    print(f"Done. [ticks={stats['ticks']}] nodes={stats['num_nodes']}, edges={stats['num_edges']}")
                    continue
                elif cmd.startswith("/rebalance"):
                    # Optionally rescale pressures to bring saturated nodes down
                    parts = cmd.split()
                    try:
                        target = float(parts[1]) if len(parts) > 1 else 50.0
                    except ValueError:
                        target = 50.0
                    brain.rebalance_pressures(target_max=target)
                    brain.save_state(STATE_PATH)
                    stats = brain.get_stats(top_k=10)
                    print(f"Rebalanced pressures to target_max={target}.")
                    print(f"[ticks={stats['ticks']}] nodes={stats['num_nodes']}, edges={stats['num_edges']}")
                    if stats["top_nodes"]:
                        print("Top nodes:")
                        for n, p in stats["top_nodes"]:
                            print(f"  - {n}: {p:.3f}")
                    continue

            if not line.strip():
                # idle tick: let the brain drift / settle
                brain.tick(steps=5)
                thoughts = brain.read_thoughts()
                if thoughts:
                    print("Idle thoughts:", thoughts)
                brain.save_state(STATE_PATH)
                continue

            reply = brain.compose_reply(line, top_k=3)
            print("Brain:", reply)

            # Self-training: let the brain "hear" its own reply and the pairing
            brain.observe_text(f"BRAIN: {reply}", strength=3.0)
            brain.observe_text(f"USER: {line} BRAIN: {reply}", strength=2.0)

            # Extra background ticks to let learning soak in
            brain.tick(steps=5)

            # Persist state after each interaction
            brain.save_state(STATE_PATH)

    except KeyboardInterrupt:
        print("\nShutting down.")