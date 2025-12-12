import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import json
import random

# --- DET 9.0 CONFIGURATION (DUALITY FIX) ---
CUBE_SIZE = 12
DOUBLE_WIDTH = 24       # Brain A (0-11) | Brain B (12-23)
DIFFUSION_RATE = 0.20   # Increased slightly to help signal travel distance
DECAY_RATE = 0.005
LEARNING_RATE = 2.0
STEPS = 1500
LOBOTOMY_STEP = 800
NEUROGENESIS_THRESHOLD = 15.0 # Lower threshold to encourage growth
REPORT_INTERVAL = 10
DATA_OUTPUT_FILE = "det_duality_fix_data.json"

class DualBrain:
    def __init__(self, size_x, size_y, size_z):
        self.sx, self.sy, self.sz = size_x, size_y, size_z
        self.pressure_a = np.zeros((size_x, size_y, size_z))
        self.pressure_b = np.zeros((size_x, size_y, size_z))
        # Initial connections: Weak but present everywhere
        self.conductance = np.random.uniform(0.1, 0.2, (size_x, size_y, size_z, 6))
        self.history = []
        self.lobotomized = False
        self.repairs = 0

    def step(self, step_count):
        # --- 1. THE TRAUMA (The Wall) ---
        if step_count == LOBOTOMY_STEP and not self.lobotomized:
            print(f"\n!!! TRAUMA EVENT AT STEP {step_count} !!!")
            print("Building a wall at x=6...")
            # We cut all flow passing through x=6.
            # This splits Brain A in half. Input (x=2) is cut off from Target (x=9).
            # The only way out is to go BACK/AROUND or ACROSS THE BRIDGE (x=11).
            self.conductance[5, :, :, 1] = 0.0 # Cut x=5 -> x=6
            self.conductance[6, :, :, 0] = 0.0 # Cut x=6 -> x=5
            self.lobotomized = True

        new_p_a = self.pressure_a.copy()
        new_p_b = self.pressure_b.copy()

        # --- 2. CALCULATE FLOW & NEUROGENESIS ---
        for x in range(1, self.sx - 1):
            for y in range(1, self.sy - 1):
                for z in range(1, self.sz - 1):
                    # Neighbors: -x, +x, -y, +y, -z, +z
                    neighbor_coords = [
                        (x-1, y, z), (x+1, y, z),
                        (x, y-1, z), (x, y+1, z),
                        (x, y, z-1), (x, y, z+1)
                    ]

                    # --- SMART NEUROGENESIS (Gradient Descent) ---
                    total_p = self.pressure_a[x, y, z]
                    
                    # If high pressure and lobotomized (or just blocked)...
                    if total_p > NEUROGENESIS_THRESHOLD:
                        # Scan neighbors for the "Path of Least Resistance" (Lowest Pressure)
                        best_target = -1
                        min_p = total_p # Start with current pressure
                        
                        for i, (nx, ny, nz) in enumerate(neighbor_coords):
                            n_p = self.pressure_a[nx, ny, nz]
                            if n_p < min_p:
                                min_p = n_p
                                best_target = i
                        
                        # If we found a lower pressure neighbor, GROW that connection
                        if best_target != -1:
                            # Growth Rule: Only grow if connection is currently weak (plasticity)
                            if self.conductance[x, y, z, best_target] < 0.5:
                                self.conductance[x, y, z, best_target] += 0.05
                                self.repairs += 1

                    # --- STANDARD FLOW ---
                    for i, (nx, ny, nz) in enumerate(neighbor_coords):
                        cond = self.conductance[x, y, z, i]
                        
                        # Signal A Flow
                        diff_a = self.pressure_a[x, y, z] - self.pressure_a[nx, ny, nz]
                        if diff_a > 0:
                            flow_a = diff_a * DIFFUSION_RATE * cond
                            new_p_a[x, y, z] -= flow_a
                            new_p_a[nx, ny, nz] += flow_a

        # --- 3. PLASTICITY (Maintain the cut) ---
        if self.lobotomized:
             self.conductance[5, :, :, 1] = 0.0
             self.conductance[6, :, :, 0] = 0.0
             
        # Atrophy (Prevent infinite growth)
        self.conductance = np.clip(self.conductance - 0.001, 0.001, 1.0)

        # --- 4. INPUTS & TARGETS ---
        input_val = 30.0 * np.sin(step_count * 0.1) + 30.0
        
        # Source at x=2
        new_p_a[2, 6, 6] = input_val
        
        # Sink/Target at x=9
        # MEASUREMENT FIX: We assume the target absorbs pressure, 
        # but we measure the pressure accumulating JUST BEFORE the sink.
        # This gives us a non-zero reading.
        target_score = new_p_a[9, 6, 6]
        new_p_a[9, 6, 6] *= 0.5 # Partial sink (resistor), not short circuit
        
        # Decay
        new_p_a *= (1 - DECAY_RATE)
        self.pressure_a = np.clip(new_p_a, 0, 100)

        # --- 5. LOGGING ---
        self.history.append({
            "step": step_count,
            "score": float(target_score),
            "repairs": int(self.repairs)
        })

        if step_count % REPORT_INTERVAL == 0:
            print(f"STEP {step_count:04d} | Score: {target_score:.2f} | New Wires: {self.repairs}")

        return self.pressure_a, self.conductance

    def save_data(self):
        print(f"\nSaving duality data to {DATA_OUTPUT_FILE}...")
        with open(DATA_OUTPUT_FILE, 'w') as f:
            json.dump(self.history, f, indent=4)
        print("Save Complete.")

# --- VISUALIZATION ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
brain = DualBrain(DOUBLE_WIDTH, CUBE_SIZE, CUBE_SIZE)

ax.set_title("DET 9.0: The Bypass (Trauma at x=6, Bridge at x=11)")
ax.set_xlabel("X (A=0-11, B=12-23)"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(0, DOUBLE_WIDTH); ax.set_ylim(0, CUBE_SIZE); ax.set_zlim(0, CUBE_SIZE)
ax.set_facecolor('#101010')
fig.patch.set_facecolor('#101010')
ax.grid(False)
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

# Visual Markers
x, y, z = np.indices((DOUBLE_WIDTH, CUBE_SIZE, CUBE_SIZE))
scatter = ax.scatter(x.flatten(), y.flatten(), z.flatten(), s=20, alpha=0.1, marker='o')

# Draw the Cut Line
ax.plot([6, 6], [0, 12], [0, 12], color='red', linestyle='-', linewidth=2, label="Trauma Zone")
# Draw the Bridge Line
ax.plot([11.5, 11.5], [0, 12], [0, 12], color='cyan', linestyle='--', linewidth=1, label="Corpus Callosum")

print("Simulating DET 9.0 Duality Fix...")
print("-" * 60)

def animate(i):
    if i == STEPS - 1:
        brain.save_data()
        
    p_a, cond = brain.step(i)
    flat_a = p_a.flatten()
    flat_cond = np.mean(cond, axis=3).flatten()
    
    colors = np.zeros((len(flat_a), 4))
    colors[:, 0] = np.clip(flat_a / 15.0, 0, 1) # Red Signal
    colors[:, 1] = np.clip(flat_cond * 0.3, 0, 0.3) # Structure
    colors[:, 3] = np.clip((flat_a/15.0) + (flat_cond*0.5), 0.0, 0.9) # Alpha

    scatter.set_color(colors)
    ax.view_init(elev=25, azim=i * 0.2)
    return [scatter]

ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=1, blit=False, repeat=False)
plt.show()