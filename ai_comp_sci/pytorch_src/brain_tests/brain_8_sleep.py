import numpy as np
import json
import matplotlib.pyplot as plt

# --- DET 9.0 CONFIGURATION (THE SLEEP TEST) ---
CUBE_SIZE = 12          
DIFFUSION_RATE = 0.15   
DECAY_RATE = 0.005      
LEARNING_RATE = 2.0     
STEPS = 1500            
SLEEP_START = 1000      # Bedtime
REPORT_INTERVAL = 10
OUTPUT_FILE = "det_sleep_data.json"

class BioCube:
    def __init__(self, size):
        self.size = size
        self.pressure_a = np.zeros((size, size, size)) 
        self.pressure_b = np.zeros((size, size, size)) 
        self.conductance = np.random.uniform(0.1, 0.2, (size, size, size, 6))
        self.history = []
        
    def step(self, step_count, dissonance=False):
        new_p_a = self.pressure_a.copy()
        new_p_b = self.pressure_b.copy()
        flow_map_total = np.zeros((self.size, self.size, self.size, 6))

        # --- 1. CALCULATE 3D FLOW ---
        for x in range(1, self.size - 1):
            for y in range(1, self.size - 1):
                for z in range(1, self.size - 1):
                    neighbors = [
                        (x-1, y, z), (x+1, y, z),
                        (x, y-1, z), (x, y+1, z),
                        (x, y, z-1), (x, y, z+1)
                    ]
                    for i, (nx, ny, nz) in enumerate(neighbors):
                        cond = self.conductance[x, y, z, i]
                        
                        # Signal A
                        diff_a = self.pressure_a[x, y, z] - self.pressure_a[nx, ny, nz]
                        if diff_a > 0:
                            flow_a = diff_a * DIFFUSION_RATE * cond
                            new_p_a[x, y, z] -= flow_a
                            new_p_a[nx, ny, nz] += flow_a
                            flow_map_total[x, y, z, i] += flow_a 

                        # Signal B
                        diff_b = self.pressure_b[x, y, z] - self.pressure_b[nx, ny, nz]
                        if diff_b > 0:
                            flow_b = diff_b * DIFFUSION_RATE * cond
                            new_p_b[x, y, z] -= flow_b
                            new_p_b[nx, ny, nz] += flow_b
                            flow_map_total[x, y, z, i] += flow_b

        # --- 2. SHARED PLASTICITY ---
        current_atrophy = 0.0 if step_count < 200 else 0.008
        self.conductance += flow_map_total * LEARNING_RATE
        self.conductance -= current_atrophy
        self.conductance = np.clip(self.conductance, 0.005, 1.0)

        # --- 3. INPUTS (AWAKE vs SLEEP) ---
        if step_count < SLEEP_START:
            # AWAKE: Inputs are Active
            if not dissonance:
                # Synchronized (Peaceful)
                input_a = 20.0 * np.sin(step_count * 0.1) + 20.0
                input_b = 20.0 * np.sin(step_count * 0.1) + 20.0
            else:
                # Dissonant (Anxious)
                input_a = 20.0 * np.sin(step_count * 0.1) + 20.0
                input_b = 20.0 * np.cos(step_count * 0.1) + 20.0
        else:
            # SLEEP: Inputs are OFF
            input_a = 0.0
            input_b = 0.0

        # Apply Inputs
        new_p_a[1, 1, 1] = input_a   
        new_p_b[10, 10, 10] = input_b 
        
        # Sinks remain active (metabolism continues)
        new_p_a[6, 6, 6] = 0.0 
        new_p_b[5, 5, 5] = 0.0 
        
        # Decay
        new_p_a *= (1 - DECAY_RATE)
        new_p_b *= (1 - DECAY_RATE)
        self.pressure_a = np.clip(new_p_a, 0, 100)
        self.pressure_b = np.clip(new_p_b, 0, 100)

        # --- 4. LOGGING ---
        center_a = float(new_p_a[5, 6, 6])
        center_b = float(new_p_b[6, 5, 5])
        score = min(center_a, center_b)
        
        self.history.append({
            "step": step_count,
            "concept_score": score,
            "vision_center": center_a,
            "sound_center": center_b,
            "mode": "Dissonance" if dissonance else "Association",
            "state": "Awake" if step_count < SLEEP_START else "Sleep"
        })

        if step_count % REPORT_INTERVAL == 0:
            state = "SLEEP" if step_count >= SLEEP_START else "AWAKE"
            mode = "DISS " if dissonance else "ASSOC"
            print(f"{mode} | {state} | Step {step_count:04d} | Score: {score:.4f}")

        return self.pressure_a, self.pressure_b

# --- RUN EXPERIMENTS ---
print(">>> STARTING SLEEP / MEMORY TEST <<<")
all_data = []

# Experiment 1: Association (Healthy Brain)
print("\n--- RUN 1: ASSOCIATIVE MEMORY ---")
cube_assoc = BioCube(CUBE_SIZE)
for i in range(STEPS):
    cube_assoc.step(i, dissonance=False)
all_data.extend(cube_assoc.history)

# Experiment 2: Dissonance (Anxious Brain)
print("\n--- RUN 2: DISSONANT MEMORY ---")
cube_diss = BioCube(CUBE_SIZE)
for i in range(STEPS):
    cube_diss.step(i, dissonance=True)
all_data.extend(cube_diss.history)

# Save Data
print(f"\nSaving sleep data to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_data, f, indent=4)
print("Save Complete.")