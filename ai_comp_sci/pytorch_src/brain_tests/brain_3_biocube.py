import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Required for 3D plotting
from mpl_toolkits.mplot3d import Axes3D 

# --- DET 8.0 CONFIGURATION (THE 3D CUBE) ---
CUBE_SIZE = 12          # 12x12x12 = 1,728 Cells
DIFFUSION_RATE = 0.15   # Lower for 3D stability (more neighbors = more flow)
DECAY_RATE = 0.005      
LEARNING_RATE = 2.0     # Aggressive wiring
STEPS = 5000            
REPORT_INTERVAL = 10

class BioCube:
    def __init__(self, size):
        self.size = size
        # 3D Pressure Fields [x, y, z]
        self.pressure_a = np.zeros((size, size, size)) # Vision (Red)
        self.pressure_b = np.zeros((size, size, size)) # Sound (Blue)
        
        # 3D Conductance Tensor: [x, y, z, 6]
        # Neighbors: Left, Right, Back, Front, Down, Up
        self.conductance = np.random.uniform(0.1, 0.2, (size, size, size, 6))
        
    def step(self, step_count):
        new_p_a = self.pressure_a.copy()
        new_p_b = self.pressure_b.copy()
        flow_map = np.zeros((self.size, self.size, self.size, 6))

        # --- 1. CALCULATE 3D FLOW ---
        # Iterate safely ignoring boundaries
        for x in range(1, self.size - 1):
            for y in range(1, self.size - 1):
                for z in range(1, self.size - 1):
                    
                    # 6 Neighbors: (-x, +x, -y, +y, -z, +z)
                    neighbors = [
                        (x-1, y, z), (x+1, y, z),
                        (x, y-1, z), (x, y+1, z),
                        (x, y, z-1), (x, y, z+1)
                    ]
                    
                    for i, (nx, ny, nz) in enumerate(neighbors):
                        cond = self.conductance[x, y, z, i]
                        
                        # --- Signal A (Vision) ---
                        diff_a = self.pressure_a[x, y, z] - self.pressure_a[nx, ny, nz]
                        if diff_a > 0:
                            flow_a = diff_a * DIFFUSION_RATE * cond
                            new_p_a[x, y, z] -= flow_a
                            new_p_a[nx, ny, nz] += flow_a
                            flow_map[x, y, z, i] += flow_a

                        # --- Signal B (Sound) ---
                        diff_b = self.pressure_b[x, y, z] - self.pressure_b[nx, ny, nz]
                        if diff_b > 0:
                            flow_b = diff_b * DIFFUSION_RATE * cond
                            new_p_b[x, y, z] -= flow_b
                            new_p_b[nx, ny, nz] += flow_b
                            flow_map[x, y, z, i] += flow_b

        # --- 2. PLASTICITY (Grace Period < 200) ---
        current_atrophy = 0.0 if step_count < 200 else 0.01
        
        self.conductance += flow_map * LEARNING_RATE
        self.conductance -= current_atrophy
        self.conductance = np.clip(self.conductance, 0.005, 1.0)

        # --- 3. INPUTS (3D Coordinates) ---
        # Vision A: Corner (2,2,2) -> Opposite Corner (10,10,10)
        input_a = 20.0 * np.sin(step_count * 0.1) + 20.0
        new_p_a[2, 2, 2] = input_a
        new_p_a[10, 10, 10] = 0.0 # Sink A
        
        # Sound B: Corner (2,10,2) -> Opposite Corner (10,2,10)
        # Crossing path on the Y-axis
        input_b = 20.0 * np.sin(step_count * 0.05) + 20.0
        new_p_b[2, 10, 2] = input_b
        new_p_b[10, 2, 10] = 0.0 # Sink B
        
        # Decay
        new_p_a *= (1 - DECAY_RATE)
        new_p_b *= (1 - DECAY_RATE)
        self.pressure_a = np.clip(new_p_a, 0, 100)
        self.pressure_b = np.clip(new_p_b, 0, 100)

        if step_count % REPORT_INTERVAL == 0:
            sink_a = new_p_a[10, 10, 9] 
            sink_b = new_p_b[10, 2, 9]
            print(f"STEP {step_count:04d} | Vis(A): {sink_a:.4f} | Snd(B): {sink_b:.4f}")

        return self.pressure_a, self.pressure_b, self.conductance

# --- 3D VISUALIZATION ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
grid = BioCube(CUBE_SIZE)

ax.set_title("DET 8.0: The Bio-Cube (3D Cortex)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_xlim(0, CUBE_SIZE)
ax.set_ylim(0, CUBE_SIZE)
ax.set_zlim(0, CUBE_SIZE)

# We use a Scatter Plot to show active "Neurons"
# We only update the colors/alpha of the points
x, y, z = np.indices((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
scatter = ax.scatter(x.flatten(), y.flatten(), z.flatten(), s=20, alpha=0.1)

print("Simulating 3D Bio-Cube...")
print("-" * 50)
print(f"{'STEP':<9} | {'VISION OUT':<12} | {'SOUND OUT':<12}")
print("-" * 50)

def animate(i):
    p_a, p_b, cond = grid.step(i)
    
    # Flatten arrays for scatter plot
    flat_a = p_a.flatten()
    flat_b = p_b.flatten()
    
    # Calculate Colors
    # Red channel = Vision
    # Blue channel = Sound
    # Green = Structure (Conductance average)
    flat_cond = np.mean(cond, axis=3).flatten()
    
    colors = np.zeros((len(flat_a), 4)) # RGBA
    colors[:, 0] = np.clip(flat_a / 20.0, 0, 1) # R
    colors[:, 2] = np.clip(flat_b / 20.0, 0, 1) # B
    colors[:, 1] = np.clip(flat_cond * 0.3, 0, 1) # G (Dim structure)
    
    # Dynamic Alpha: Only show active cells
    # If a cell has no pressure and no conductance, make it invisible
    activity = colors[:, 0] + colors[:, 2] + flat_cond
    colors[:, 3] = np.clip(activity, 0.01, 0.8) # Alpha
    
    scatter.set_color(colors)
    
    # Slowly rotate the cube for cinematic effect
    ax.view_init(elev=20, azim=i * 0.5)
    
    return [scatter]

ani = animation.FuncAnimation(fig, animate, frames=STEPS, interval=1, blit=False)
plt.show()