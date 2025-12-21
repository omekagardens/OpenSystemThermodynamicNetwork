import networkx as nx
import operator
import json
import math
import random
import time

# --- BIO-DET CONFIGURATION ---

# Physics
DIFFUSION_RATE      = 0.15   # Flow speed
DECAY_RATE          = 0.05   # Forgetfulness
LEARNING_RATE       = 0.01   # Hebbian plasticity
MAX_PRESSURE        = 100.0  # Clamp ceiling

# Biology (New)
FATIGUE_RATE        = 0.1    # How fast a concept gets "boring"
RECOVERY_RATE       = 0.02   # How fast interest recovers
FATIGUE_SCALE       = 5.0    # How strongly fatigue suppresses pressure

# Structure
ANCHORS             = ["GOOD", "BAD", "SELF", "USER"]

class BioDETBrain:
    """
    A biological-style semantic network.
    differs from standard graphs by having 'Fatigue' (boredom) 
    and 'Valence' (emotional grounding).
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.node_pressure = {}      # Activity (Voltage)
        self.node_fatigue = {}       # Adaptation (Boredom)
        self.edge_conductance = {}   # Synaptic Weight
        self.tick_index = 0
        
        # Initialize the "Limbic System" (Anchors)
        for anchor in ANCHORS:
            self.add_concept(anchor)
            # Anchors are harder to fatigue (instincts are persistent)
            self.node_fatigue[anchor] = -10.0 

    def add_concept(self, name: str):
        name = name.upper() # Mentalese is distinct from English
        if name not in self.graph:
            self.graph.add_node(name)
            self.node_pressure[name] = 0.0
            self.node_fatigue[name] = 0.0

    def associate(self, n1: str, n2: str, weight: float = 1.0):
        n1, n2 = n1.upper(), n2.upper()
        self.add_concept(n1)
        self.add_concept(n2)
        
        if not self.graph.has_edge(n1, n2):
            self.graph.add_edge(n1, n2)
        
        key = tuple(sorted((n1, n2)))
        # Sigmoidal update to prevent explosion
        current = self.edge_conductance.get(key, 0.5)
        self.edge_conductance[key] = (current + weight) / 2.0

    # --- THE PHYSICS ENGINE ---

    def tick(self, steps=1):
        for _ in range(steps):
            self._diffusion_step()
            self._biology_step()
            self.tick_index += 1

    def _diffusion_step(self):
        """Standard pressure diffusion (The 'Thinking' Flow)"""
        fluxes = {n: 0.0 for n in self.graph.nodes}
        
        # Calculate Flow
        for (u, v) in self.graph.edges:
            key = tuple(sorted((u, v)))
            cond = self.edge_conductance.get(key, 1.0)
            
            p_u = self.node_pressure[u]
            p_v = self.node_pressure[v]
            
            # Simple diffusion
            flow = (p_u - p_v) * DIFFUSION_RATE * cond
            fluxes[u] -= flow
            fluxes[v] += flow

            # Hebbian Learning: If both fire, strengthen bond
            if abs(p_u) > 10 and abs(p_v) > 10:
                self.edge_conductance[key] = min(5.0, cond + LEARNING_RATE)

        # Apply Flow
        for n in self.graph.nodes:
            self.node_pressure[n] += fluxes[n]
            self.node_pressure[n] *= (1.0 - DECAY_RATE) # Natural decay
            
            # Hard Clamp
            self.node_pressure[n] = max(-MAX_PRESSURE, min(MAX_PRESSURE, self.node_pressure[n]))

    def _biology_step(self):
        """The 'Feeling' Control: Fatigue and Refractory Periods"""
        for n in self.graph.nodes:
            # 1. Fatigue Accumulation
            # If a node is screaming (high pressure), it gets tired.
            if self.node_pressure[n] > 20.0:
                self.node_fatigue[n] += FATIGUE_RATE
            
            # 2. Fatigue Recovery (The 'Boredom' Decay)
            # Nodes slowly recover interest if left alone
            if self.node_fatigue[n] > 0:
                self.node_fatigue[n] -= RECOVERY_RATE
                if self.node_fatigue[n] < 0: self.node_fatigue[n] = 0

    # --- THE GHOST INTERFACE ---

    def get_effective_state(self, top_k=5):
        """
        Returns the 'Conscious' thoughts.
        Critical: Applies Fatigue filter. A high-pressure node with 
        high fatigue is INVISIBLE to the interface (The Topic Lock Fix).
        """
        candidates = []
        for n, p in self.node_pressure.items():
            fatigue = self.node_fatigue.get(n, 0.0)
            
            # The Formula: Effective = Pressure / (1 + Fatigue * Scale)
            effective_p = p / (1.0 + (fatigue * FATIGUE_SCALE))
            
            if effective_p > 5.0: # Minimum awareness threshold
                candidates.append((n, effective_p))
        
        # Sort by effective pressure
        candidates.sort(key=operator.itemgetter(1), reverse=True)
        return candidates[:top_k]

    def inject_input(self, concept_list, strength=50.0):
        """Inject sensory data"""
        for c in concept_list:
            c = c.upper()
            self.add_concept(c)
            # Input fights against fatigue instantly (Novelty Bonus)
            self.node_pressure[c] += strength
            self.node_fatigue[c] = max(0, self.node_fatigue[c] - 2.0)

# --- THE LLM BRIDGE (Interface Layer) ---

class LlamaInterface:
    def __init__(self, model_path=None):
        self.mock_mode = True
        self.llm = None
        
        if model_path:
            try:
                from llama_cpp import Llama
                print(f"Loading Llama from {model_path}...")
                self.llm = Llama(model_path=model_path, verbose=False)
                self.mock_mode = False
            except ImportError:
                print("!! llama-cpp-python not found. Running in MOCK mode. !!")
            except Exception as e:
                print(f"!! Error loading model: {e}. Running in MOCK mode. !!")

    def extract_concepts(self, user_text):
        """
        Perception: Text -> Triples -> Concepts
        """
        if self.mock_mode:
            # Simple mock keyword extractor for testing
            words = user_text.split()
            return [w.strip(".,?!") for w in words if len(w) > 3]
        
        # Real Llama Extraction Prompt
        prompt = f"""
        Extract the core entities and emotions from: "{user_text}"
        Return ONLY a JSON list of strings (uppercase).
        Example: ["CAR", "FAST", "EXCITEMENT"]
        JSON:
        """
        output = self.llm(prompt, max_tokens=64, stop=["\n"], echo=False)
        try:
            # Heuristic cleanup of Llama output
            text = output['choices'][0]['text'].strip()
            # If model forgets JSON, just fallback to text
            if "[" in text: 
                return json.loads(text[text.find("["):text.find("]")+1])
            else:
                return [w.upper() for w in text.split() if len(w)>3]
        except:
            return []

    def generate_reply(self, user_text, brain_state):
        """
        Action: (Context + Feeling) -> Text
        """
        # 1. Convert brain state to 'Context' string
        thoughts = [t[0] for t in brain_state]
        
        # 2. Check Emotional Anchors
        sentiment = "NEUTRAL"
        if "GOOD" in thoughts: sentiment = "POSITIVE"
        if "BAD" in thoughts: sentiment = "NEGATIVE"
        
        # 3. Construct the 'Ghost' Prompt
        system_instruction = f"""
        You are an AI assistant with a bio-digital brain.
        Current internal state: {thoughts}
        Emotional Valence: {sentiment}
        
        Instruction: Respond to the user.
        If your state contains random words, try to connect them metaphorically.
        If 'BAD' is active, be skeptical or sad.
        If 'GOOD' is active, be enthusiastic.
        """
        
        prompt = f"System: {system_instruction}\nUser: {user_text}\nAssistant:"
        
        if self.mock_mode:
            return f"[MOCK LLM] I feel {sentiment} about {thoughts}. (User said: {user_text})"
        
        # 4. (Advanced) Logit Bias Control
        # This forces the LLM to focus on the active concepts
        # We map the concept words to a slight probability boost
        logit_bias = {}
        # Note: In a real implementation, you'd need the tokenizer to get IDs
        # logit_bias = { self.llm.tokenize(w.encode("utf-8"))[0] : 2.0 for w in thoughts }
        
        output = self.llm(
            prompt, 
            max_tokens=128, 
            temperature=0.7,
            # logit_bias=logit_bias # Uncomment if you map tokens correctly
        )
        return output['choices'][0]['text'].strip()

# --- REPL ---

def main():
    # SETUP
    brain = BioDETBrain()
    
    # Point this to your actual .gguf file if you have one
    # e.g. "models/llama-3-8b-instruct.Q4_K_M.gguf"
    llama = LlamaInterface(model_path=None) 

    # Seed some initial 'Instincts'
    brain.associate("USER", "GOOD", weight=2.0) # Innately likes the user
    brain.associate("PAIN", "BAD", weight=5.0)  # Innately hates pain
    
    print(">>> BioDET Brain Online. (Type /stats to see internal pressures) <<<")
    
    while True:
        try:
            user_input = input("\nUSER: ")
            if not user_input: continue
            
            if user_input.startswith("/"):
                # Debug Commands
                if user_input == "/stats":
                    print("\n--- BRAIN STATE ---")
                    # Show top raw pressures
                    raw = sorted(brain.node_pressure.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"Top Pressure (Raw): {[(k, round(v,1)) for k,v in raw]}")
                    
                    # Show fatigue
                    tired = sorted(brain.node_fatigue.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"Top Fatigue (Boredom): {[(k, round(v,2)) for k,v in tired]}")
                    
                    # Show what the LLM actually sees
                    effective = brain.get_effective_state()
                    print(f"Conscious Thoughts: {effective}")
                    continue
                    
                if user_input == "/sleep":
                    print("Consolidating memories...")
                    # TODO: Implement pruning here
                    brain.tick(20) # Fast forward time
                    print("Brain rested.")
                    continue

            # 1. Perception
            concepts = llama.extract_concepts(user_input)
            print(f"(extracted: {concepts})")
            
            # 2. Injection
            brain.inject_input(concepts)
            
            # 3. Processing (The 'Time' step)
            # We tick multiple times to let the pressure flow and fatigue settle
            brain.tick(steps=10)
            
            # 4. Readout (The 'Ghost' State)
            active_thoughts = brain.get_effective_state(top_k=4)
            
            # 5. Action
            response = llama.generate_reply(user_input, active_thoughts)
            print(f"BRAIN: {response}")
            
            # 6. Feedback Loop (Hearing itself)
            # The brain processes its own output, reinforcing the path
            self_concepts = llama.extract_concepts(response)
            brain.inject_input(self_concepts, strength=10.0) # Weaker than user input
            
        except KeyboardInterrupt:
            print("\nShutting down.")
            break

if __name__ == "__main__":
    main()
