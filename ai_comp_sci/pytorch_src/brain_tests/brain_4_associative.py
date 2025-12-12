import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import json
import os

# --- DET 9.0 CONFIGURATION (THE ASSOCIATIVE WEB) ---
CUBE_SIZE = 12          
DIFFUSION_RATE = 0.15   
DECAY_RATE = 0.005      
LEARNING_RATE = 2.0     
STEPS = 1500            # Total run time
REPORT_INTERVAL = 10
DATA_OUTPUT_FILE = "det_simulation_data.json"

class BioCube:
    def __init__(self, size):
        self.size = size
        # Two separate pressure fields sharing one physical network
        self.pressure_a = np.zeros((size, size, size)) # Vision (Red)
        self.pressure_b = np.zeros((size, size, size)) # Sound (Blue)
        # The Physical Wires (Conductance)
        self.conductance = np.random.uniform(0.1, 0.2, (size, size, size, 6))
        # Data recording
        self.history = []
        
    def step(self, step_count):
        new_p_a = self.pressure_a.copy()
        new_p_b = self.pressure_b.copy()
        # Track total flow of BOTH signals through pipes
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
                        
                        # Signal A (Vision)
                        diff_a = self.pressure_a[x, y, z] - self.pressure_a[nx, ny, nz]
                        if diff_a > 0:
                            flow_a = diff_a * DIFFUSION_RATE * cond
                            new_p_a[x, y, z] -= flow_a
                            new_p_a[nx, ny, nz] += flow_a
                            flow_map_total[x, y, z, i] += flow_a 

                        # Signal B (Sound)
                        diff_b = self.pressure_b[x, y, z] - self.pressure_b[nx, ny, nz]
                        if diff_b > 0:
                            flow_b = diff_b * DIFFUSION_RATE * cond
                            new_p_b[x, y, z] -= flow_b
                            new_p_b[nx, ny, nz] += flow_b
                            flow_map_total[x, y, z, i] += flow_b

        # --- 2. SHARED PLASTICITY ---
        # Atrophy starts after step 200 to allow initial growth
        current_atrophy = 0.0 if step_count < 200 else 0.008
        self.conductance += flow_map_total * LEARNING_RATE
        self.conductance -= current_atrophy
        self.conductance = np.clip(self.conductance, 0.005, 1.0)

        # --- 3. INPUTS & TARGETS ---
        input_val = 20.0 * np.sin(step_count * 0.1) + 20.0
        new_p_a[1, 1, 1] = input_val   # Vision Input
        new_p_b[10, 10, 10] = input_val # Sound Input
        
        # Sinks (Targets)
        new_p_a[6, 6, 6] = 0.0 
        new_p_b[5, 5, 5] = 0.0 
        
        # Decay
        new_p_a *= (1 - DECAY_RATE)
        new_p_b *= (1 - DECAY_RATE)
        self.pressure_a = np.clip(new_p_a, 0, 100)
        self.pressure_b = np.clip(new_p_b, 0, 100)

        # --- 4. DATA LOGGING ---
        center_a = float(new_p_a[5, 6, 6])
        center_b = float(new_p_b[6, 5, 5])
        score = min(center_a, center_b)
        
        # Log every step internally
        self.history.append({
            "step": step_count,
            "vision_center": center_a,
            "sound_center": center_b,
            "concept_score": score
        })

        if step_count % REPORT_INTERVAL == 0:
            print(f"STEP {step_count:04d} | Vis: {center_a:.2f} | Snd: {center_b:.2f} | SCORE: {score:.2f}")

        return self.pressure_a, self.pressure_b, self.conductance

    def save_data(self):
        print(f"\nSaving analysis data to {DATA_OUTPUT_FILE}...")
        with open(DATA_OUTPUT_FILE, 'w') as f:
            json.dump(self.history, f, indent=4)
        print("Save Complete.")

# --- 3D VISUALIZATION ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
grid = BioCube(CUBE_SIZE)

ax.set_title("DET 9.0: The Associative Web")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(0, CUBE_SIZE); ax.set_ylim(0, CUBE_SIZE); ax.set_zlim(0, CUBE_SIZE)
ax.set_facecolor('#101010') # Dark gray background
fig.patch.set_facecolor('#101010')
ax.grid(False)

# Fixed Axis Color
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

# --- VISUAL REFERENCE CAGE ---
# Draws a static white box so we can see the network "shrink" (optimize) relative to fixed space.
edges = [
    [[0, CUBE_SIZE], [0, 0], [0, 0]], [[0, CUBE_SIZE], [CUBE_SIZE, CUBE_SIZE], [0, 0]],
    [[0, CUBE_SIZE], [0, 0], [CUBE_SIZE, CUBE_SIZE]], [[0, CUBE_SIZE], [CUBE_SIZE, CUBE_SIZE], [CUBE_SIZE, CUBE_SIZE]],
    [[0, 0], [0, CUBE_SIZE], [0, 0]], [[CUBE_SIZE, CUBE_SIZE], [0, CUBE_SIZE], [0, 0]],
    [[0, 0], [0, CUBE_SIZE], [CUBE_SIZE, CUBE_SIZE]], [[CUBE_SIZE, CUBE_SIZE], [0, CUBE_SIZE], [CUBE_SIZE, CUBE_SIZE]],
    [[0, 0], [0, 0], [0, CUBE_SIZE]], [[CUBE_SIZE, CUBE_SIZE], [0, 0], [0, CUBE_SIZE]],
    [[0, 0], [CUBE_SIZE, CUBE_SIZE], [0, CUBE_SIZE]], [[CUBE_SIZE, CUBE_SIZE], [CUBE_SIZE, CUBE_SIZE], [0, CUBE_SIZE]],
]
for edge in edges:
    ax.plot(edge[0], edge[1], edge[2], color='white', alpha=0.15, linewidth=0.8)

# Initialize Scatter
x, y, z = np.indices((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
scatter = ax.scatter(x.flatten(), y.flatten(), z.flatten(), s=40, alpha=0.1, marker='o')

print("Simulating DET 9.0 Associative Web...")
print("-" * 60)

def animate(i):
    # Stop condition
    if i == STEPS - 1:
        grid.save_data()
        
    p_a, p_b, cond = grid.step(i)
    flat_a = p_a.flatten()
    flat_b = p_b.flatten()
    flat_cond = np.mean(cond, axis=3).flatten()
    
    # --- COLOR MIXING ---
    colors = np.zeros((len(flat_a), 4))
    
    # Brightness Normalization
    norm_a = np.clip(flat_a / 12.0, 0, 1) # Red (Vision)
    norm_b = np.clip(flat_b / 12.0, 0, 1) # Blue (Sound)
    
    colors[:, 0] = norm_a 
    colors[:, 2] = norm_b 
    colors[:, 1] = np.clip(flat_cond * 0.4, 0, 0.4) # Green (Structure/Wire)

    # Visibility (Alpha)
    # This logic hides nodes that have no pressure AND no structure (Atrophy visualization)
    activity = norm_a + norm_b + (flat_cond * 0.6)
    colors[:, 3] = np.clip(activity, 0.0, 0.85)

    scatter.set_color(colors)
    ax.view_init(elev=20, azim=i * 0.25)
    return [scatter]

ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=1, blit=False, repeat=False)
plt.show()