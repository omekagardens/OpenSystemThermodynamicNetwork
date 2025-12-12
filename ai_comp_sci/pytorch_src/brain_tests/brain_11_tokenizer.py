import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import json

# --- DET 9.0 CONFIGURATION (THE TRANSLATOR FIX) ---
CUBE_SIZE = 12
DIFFUSION_RATE = 0.20
DECAY_RATE = 0.05       
LEARNING_RATE = 0.0     
STEPS = 500
REPORT_INTERVAL = 10
DATA_OUTPUT_FILE = "det_language_fix_data.json"

class DETTokenizer:
    def __init__(self):
        # The Semantic Square
        self.vocab = {
            "KING":  (3, 3, 6),
            "MAN":   (3, 9, 6),  
            "WOMAN": (9, 9, 6),  
            "QUEEN": (9, 3, 6)   
        }

    def get_coords(self, word):
        return self.vocab.get(word.upper())

class BioCube:
    def __init__(self, size):
        self.size = size
        self.pressure = np.zeros((size, size, size))
        self.conductance = np.ones((size, size, size, 6)) * 0.5
        self.tokenizer = DETTokenizer()
        self.history = []

    def inject_word(self, pressure_array, word, value):
        coords = self.tokenizer.get_coords(word)
        if coords:
            # Inject directly into the active array
            pressure_array[coords] += value

    def step(self, step_count):
        new_p = self.pressure.copy()

        # --- 1. CALCULATE FLOW ---
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
                        diff = self.pressure[x, y, z] - self.pressure[nx, ny, nz]
                        flow = diff * DIFFUSION_RATE * cond
                        new_p[x, y, z] -= flow
                        new_p[nx, ny, nz] += flow

        # --- 2. INPUTS (The Fix) ---
        # Injecting into new_p ensures the energy persists to the next step
        self.inject_word(new_p, "KING", 10.0)   # + Source
        self.inject_word(new_p, "WOMAN", 10.0)  # + Source
        self.inject_word(new_p, "MAN", -10.0)   # - Sink

        # --- 3. DECAY ---
        new_p *= (1 - DECAY_RATE)
        self.pressure = new_p 

        # --- 4. LOGGING ---
        q_coords = self.tokenizer.get_coords("QUEEN")
        k_coords = self.tokenizer.get_coords("KING")
        
        queen_p = self.pressure[q_coords]
        king_p = self.pressure[k_coords] 
        
        self.history.append({
            "step": step_count,
            "queen_pressure": float(queen_p),
            "king_pressure": float(king_p)
        })

        if step_count % REPORT_INTERVAL == 0:
            print(f"STEP {step_count:04d} | Target (QUEEN): {queen_p:.2f} | Source (KING): {king_p:.2f}")

        return self.pressure

    def save_data(self):
        print(f"\nSaving language data to {DATA_OUTPUT_FILE}...")
        with open(DATA_OUTPUT_FILE, 'w') as f:
            json.dump(self.history, f, indent=4)
        print("Save Complete.")

# --- 3D VISUALIZATION ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
brain = BioCube(CUBE_SIZE)
tokenizer = DETTokenizer()

ax.set_title("DET 9.0: Semantic Algebra (King - Man + Woman)")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
ax.set_xlim(0, CUBE_SIZE); ax.set_ylim(0, CUBE_SIZE); ax.set_zlim(0, CUBE_SIZE)
ax.set_facecolor('#101010')
fig.patch.set_facecolor('#101010')
ax.grid(False)
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))

x, y, z = np.indices((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
scatter = ax.scatter(x.flatten(), y.flatten(), z.flatten(), s=50, alpha=0.1, marker='s')

for word, coords in tokenizer.vocab.items():
    ax.text(coords[0], coords[1], coords[2], word, color='white', fontsize=10, fontweight='bold')

print("Simulating DET Language Processing (Fixed)...")
print("-" * 60)

def animate(i):
    if i == STEPS - 1:
        brain.save_data()
        
    p = brain.step(i)
    flat_p = p.flatten()
    
    colors = np.zeros((len(flat_p), 4))
    
    # Normalize for display (Max theoretical is ~200 with decay)
    norm_p = flat_p / 100.0
    
    colors[:, 0] = np.clip(norm_p, 0, 1) # Red (+)
    colors[:, 2] = np.clip(-norm_p, 0, 1) # Blue (-)
    colors[:, 3] = np.clip(np.abs(norm_p)*2.0, 0.0, 0.8)

    scatter.set_color(colors)
    ax.view_init(elev=90, azim=-90) 
    return [scatter]

ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=1, blit=False, repeat=False)
plt.show()