import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# --- DET 6.0 CONFIGURATION (NEUROPLASTICITY TEST) ---
GRID_SIZE = 20
DIFFUSION_RATE = 0.5    
DECAY_RATE = 0.01       
LEARNING_RATE = 0.5     # High Plasticity to allow rapid regrowth
STEPS = 2000            
REPORT_INTERVAL = 10
LESION_STEP = 800       # When the surgery happens

class BioGrid:
    def __init__(self, size):
        self.size = size
        self.pressure = np.zeros((size, size))
        # Robust initialization
        self.conductance = np.random.uniform(0.1, 0.3, (size, size, 4))
        
    def step(self, step_count):
        new_pressure = self.pressure.copy()
        flow_map = np.zeros((self.size, self.size, 4))
        total_flow_volume = 0.0

        # --- 1. CALCULATE FLOW ---
        for r in range(1, self.size - 1):
            for c in range(1, self.size - 1):
                current_p = self.pressure[r, c]
                neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
                
                for i, (dr, dc) in enumerate(neighbors):
                    nr, nc = r + dr, c + dc
                    neighbor_p = self.pressure[nr, nc]
                    diff = current_p - neighbor_p
                    
                    if diff > 0:
                        cond = self.conductance[r, c, i]
                        # Tanh saturation
                        flow = np.tanh(diff) * DIFFUSION_RATE * cond
                        
                        new_pressure[r, c] -= flow
                        new_pressure[nr, nc] += flow
                        flow_map[r, c, i] = flow
                        total_flow_volume += flow

        # --- 2. SURGICAL INTERVENTION (THE LESION) ---
        # Define the "Scar Tissue" zone (Center of grid)
        center_start = self.size // 2 - 3
        center_end = self.size // 2 + 3
        
        if step_count == LESION_STEP:
            print("\n!!! SURGICAL INTERVENTION: NEURAL LESION APPLIED !!!\n")
            
        if step_count >= LESION_STEP:
            # Continuously kill connections in the center (The Scar)
            # This forces the AI to route AROUND it, not through it.
            self.conductance[center_start:center_end, center_start:center_end, :] = 0.0

        # --- 3. DYNAMIC ATROPHY ---
        if step_count < 400:
            current_atrophy = 0.0001 
        else:
            ramp = (step_count - 400) * 0.00002
            current_atrophy = 0.0001 + ramp
            
        current_atrophy = min(current_atrophy, 0.01)

        # --- 4. PLASTICITY ---
        self.conductance += flow_map * LEARNING_RATE
        self.conductance -= current_atrophy
        
        # Clamp (Min 0.005 ensures micro-capillaries survive to support regrowth)
        self.conductance = np.clip(self.conductance, 0.005, 1.0)

        # --- 5. BOUNDARY CONDITIONS (HEARTBEAT) ---
        # Diastolic Floor: Input pulses between 2.0 and 10.0
        pulse = 4.0 * np.sin(step_count * 0.1) + 6.0 
        new_pressure[5, 5] = pulse 
        
        sink_pressure = new_pressure[15, 15] 
        new_pressure[15, 15] = 0.0 
        
        new_pressure *= (1 - DECAY_RATE)
        new_pressure = np.clip(new_pressure, 0, 100.0)
        self.pressure = new_pressure

        if step_count % REPORT_INTERVAL == 0:
            self.report(step_count, sink_pressure, current_atrophy, pulse)

        return self.pressure, np.mean(self.conductance, axis=2), pulse

    def report(self, step, sink_val, atrophy_val, input_val):
        total_energy = np.sum(self.pressure)
        flat_cond = self.conductance.flatten()
        dead_pipes = np.sum(flat_cond < 0.02)
        total_pipes = flat_cond.size
        sparsity = (dead_pipes / total_pipes) * 100
        
        status = "HEALTHY" if step < LESION_STEP else "DAMAGED"
        
        print(f"STEP {step:04d} [{status}] | "
              f"In: {input_val:4.1f} | "
              f"Sink: {sink_val:6.4f} | "
              f"Sparse: {sparsity:4.1f}% | "
              f"Rent: {atrophy_val:.5f}")

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
grid = BioGrid(GRID_SIZE)

im1 = ax1.imshow(grid.pressure, cmap='inferno', vmin=0, vmax=10)
ax1.set_title("Pressure Field")
im2 = ax2.imshow(np.mean(grid.conductance, axis=2), cmap='Greens', vmin=0, vmax=1.0)
ax2.set_title("Neural Structure")

print("-" * 95)
print(f"{'STEP':<15} | {'INPUT':<8} | {'SINK':<11} | {'SPARSE':<10} | {'RENT'}")
print("-" * 95)

def animate(i):
    pressure_data, cond_data, current_pulse = grid.step(i)
    
    im1.set_array(pressure_data)
    im2.set_array(cond_data)
    
    # Visual Status Update
    status_icon = "❤️" if i < LESION_STEP else "💔 RECOVERING..."
    if i > LESION_STEP + 300: status_icon = "❤️‍🩹 HEALED"
    
    fig.suptitle(f"DET 6.0 Neuroplasticity: {status_icon} (Step {i})", fontsize=16)
    
    return [im1, im2]

ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=1, blit=False)
plt.show()