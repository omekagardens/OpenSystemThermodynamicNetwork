import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- DET 7.2 CONFIGURATION (STABLE PHYSICS) ---
GRID_SIZE = 20
DIFFUSION_RATE = 0.2    # <--- LOWERED for stability (No more crashes)
DECAY_RATE = 0.001      # <--- LOWERED so signals reach the other side
LEARNING_RATE = 2.0     # <--- BOOSTED: Pipes adapt instantly
STEPS = 1500            
REPORT_INTERVAL = 10

class DualBioGrid:
    def __init__(self, size):
        self.size = size
        self.pressure_a = np.zeros((size, size)) 
        self.pressure_b = np.zeros((size, size))
        self.conductance = np.random.uniform(0.2, 0.4, (size, size, 4))
        
    def step(self, step_count):
        # Temp buffers
        new_p_a = self.pressure_a.copy()
        new_p_b = self.pressure_b.copy()
        
        # Track total flow through every pipe
        flow_map_total = np.zeros((self.size, self.size, 4))

        # --- 1. CALCULATE FLOW (Thermodynamics) ---
        for r in range(1, self.size - 1):
            for c in range(1, self.size - 1):
                neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
                
                for i, (dr, dc) in enumerate(neighbors):
                    nr, nc = r + dr, c + dc
                    cond = self.conductance[r, c, i]
                    
                    # --- SIGNAL A (Red/Vision) ---
                    diff_a = self.pressure_a[r, c] - self.pressure_a[nr, nc]
                    if diff_a > 0:
                        # Standard diffusion
                        flow_a = diff_a * DIFFUSION_RATE * cond
                        new_p_a[r, c] -= flow_a
                        new_p_a[nr, nc] += flow_a
                        flow_map_total[r, c, i] += flow_a

                    # --- SIGNAL B (Blue/Sound) ---
                    diff_b = self.pressure_b[r, c] - self.pressure_b[nr, nc]
                    if diff_b > 0:
                        flow_b = diff_b * DIFFUSION_RATE * cond
                        new_p_b[r, c] -= flow_b
                        new_p_b[nr, nc] += flow_b
                        flow_map_total[r, c, i] += flow_b

        # --- 2. PLASTICITY (With Grace Period) ---
        if step_count < 200:
            current_atrophy = 0.0 # Free Growth
        else:
            current_atrophy = 0.01 # Rent is due

        self.conductance += flow_map_total * LEARNING_RATE
        self.conductance -= current_atrophy
        self.conductance = np.clip(self.conductance, 0.01, 1.0) # Floor 0.01 prevents death

        # --- 3. INPUTS (High Pressure) ---
        # A: Top Left -> Bottom Right
        input_a = 15.0 * np.sin(step_count * 0.1) + 20.0 
        new_p_a[2, 2] = input_a 
        new_p_a[17, 17] = 0.0 
        
        # B: Top Right -> Bottom Left
        input_b = 15.0 * np.sin(step_count * 0.05) + 20.0 
        new_p_b[2, 17] = input_b
        new_p_b[17, 2] = 0.0 
        
        # Decay
        new_p_a *= (1 - DECAY_RATE)
        new_p_b *= (1 - DECAY_RATE)
        
        self.pressure_a = np.clip(new_p_a, 0, 100)
        self.pressure_b = np.clip(new_p_b, 0, 100)

        # --- TELEMETRY ---
        if step_count % REPORT_INTERVAL == 0:
            sink_a = new_p_a[17, 16] 
            sink_b = new_p_b[16, 2]  
            print(f"STEP {step_count:04d} | A(Vis): {sink_a:.4f} | B(Snd): {sink_b:.4f}")

        return self.pressure_a, self.pressure_b, self.conductance

# --- VISUALIZATION ---
fig, ax = plt.subplots(figsize=(8, 8))
grid = DualBioGrid(GRID_SIZE)
img = ax.imshow(np.zeros((GRID_SIZE, GRID_SIZE, 3)), interpolation='nearest')
ax.set_title("DET 7.2: Stable Dual-Signal Brain")

print("-" * 50)
print(f"{'STEP':<9} | {'VISION OUT':<12} | {'SOUND OUT':<12}")
print("-" * 50)

def animate(i):
    p_a, p_b, cond = grid.step(i)
    # Visualization Normalization
    vis_a = np.clip(p_a / 20.0, 0, 1) 
    vis_b = np.clip(p_b / 20.0, 0, 1) 
    vis_c = np.clip(np.mean(cond, axis=2), 0, 1) 
    im_display = np.stack([vis_a, vis_c * 0.3, vis_b], axis=-1)
    img.set_data(im_display)
    return [img]

ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=1, blit=False)
plt.show()