from det_core import DETConfig, DETSystem
from det_topology import DETTopology
from det_labs import DETLab

# 1. Configure Physics
config = DETConfig(
    phi_res=3.0,
    noise_level=0.01,
    threshold_stress=3.0
)

# 2. Build Ecosystem (Forest)
adj_E, adj_I, roots, leaves = DETTopology.create_fungal_bridge(nodes_per_plant=20)
plant_A_root, plant_B_root = roots
plant_A_leaf, plant_B_leaf = leaves

nodes_per_plant = 20
meta = DETTopology.bridge_metadata(nodes_per_plant, roots, leaves)

# 3. Initialize System
forest = DETSystem(num_nodes=40, config=config)
forest.adj_energy = adj_E
forest.adj_info = adj_I

# Give roots access to soil (a is a per-node gating/control vector)
forest.a[:] = 0.0
forest.a[plant_A_root] = 1.0
forest.a[plant_B_root] = 1.0

# Wake up the ecosystem (so results reflect propagation, not just 2 live nodes)
for i in range(forest.N):
    forest.activate_node(i)

# 4. Define Custom Logic (The "API Hook")
# Let's say we want to bite Plant A at Tick 20
def bite_schedule(sim):
    if sim.tick == 20:
        DETLab.inject_chaos_bite(sim, plant_A_leaf, force=5.0)

# Register the hook
forest.register_hook('pre_step', bite_schedule)

# 5. Run Experiment
print(f"Setup: Plant A Leaf={plant_A_leaf}, Plant B Root={plant_B_root}")
history = DETLab.run_headless(forest, steps=60, log_freq=5, meta=meta)

# 6. Verify Inter-Species Comm
print("\n--- RESULTS ANALYSIS ---")
plant_b_indices = range(20, 40)
locked_b = sum(1 for i in plant_b_indices if forest.g_lateral[i] < 0.5)
print(f"Plant B Locked Nodes: {locked_b}")
first_lock = None
try:
    first_lock = history[-1].get("first_lock_B_tick", None)
except Exception:
    first_lock = None
print(f"Plant B first-lock tick: {first_lock}")
if locked_b > 2:
    print("SUCCESS: Fungal Bridge transmitted warning signal!")
else:
    print("FAIL: Signal did not cross.")