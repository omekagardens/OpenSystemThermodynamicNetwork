import torch
import torch.nn.functional as F
import numpy as np

class DETSystemV4:
    def __init__(self, max_nodes=100, phi_res=1.0, dt=0.01, noise_level=0.01):
        self.max_nodes = max_nodes
        self.dt = dt
        self.phi_res = phi_res
        self.noise_level = noise_level 
        
        # --- Pre-allocate Memory (The "Void") ---
        self.F = torch.zeros(max_nodes) 
        self.sigma = torch.ones(max_nodes) * 0.5
        self.a = torch.zeros(max_nodes) 
        self.I = torch.zeros(max_nodes) 
        
        # Control
        self.g_lateral = torch.ones(max_nodes) 
        self.danger_mask = torch.zeros(max_nodes)
        self.scream_counter = torch.zeros(max_nodes)
        
        # --- NEW: Life Mask ---
        # 0 = Dead/Dormant, 1 = Alive
        self.alive = torch.zeros(max_nodes)
        
        # Physics Constants
        self.gamma = 0.02          
        self.eta_sigma = 0.1       
        self.alpha_E = 1.0
        self.metabolic_rate = 0.01 
        
        # Topology (Pre-allocated)
        self.adj_energy = torch.zeros(max_nodes, max_nodes)
        self.adj_info = torch.zeros(max_nodes, max_nodes)
        
        self.telemetry = {
            "tick": 0, "alive_count": 0, "total_energy": 0.0, "births": 0, "deaths": 0
        }

    def activate_node(self, idx, energy, conductivity, soil_access=0.0):
        """Spawns a node from the void."""
        self.alive[idx] = 1.0
        self.F[idx] = energy
        self.sigma[idx] = conductivity
        self.a[idx] = soil_access
        self.I[idx] = 0.0
        self.g_lateral[idx] = 1.0

    def kill_node(self, idx):
        """Sends a node back to the void."""
        self.alive[idx] = 0.0
        self.F[idx] = 0.0
        self.I[idx] = 0.0
        self.a[idx] = 0.0 # Lose soil access if you die
        # Sever connections
        self.adj_energy[idx, :] = 0
        self.adj_energy[:, idx] = 0
        self.adj_info[idx, :] = 0
        self.adj_info[:, idx] = 0

    def run_evolutionary_dynamics(self):
        """The Cycle of Life and Death."""
        births = 0
        deaths = 0
        
        # 1. DEATH (Starvation)
        # Find living nodes with <= 0 Energy
        starving_indices = (self.alive == 1) & (self.F <= 0.001)
        dead_indices = torch.nonzero(starving_indices).flatten()
        
        for idx in dead_indices:
            self.kill_node(idx)
            deaths += 1

        # 2. BIRTH (Mitosis)
        # Find living nodes with Surplus Energy (> 5.0)
        repro_indices = (self.alive == 1) & (self.F > 5.0)
        parents = torch.nonzero(repro_indices).flatten()
        
        # Find dormant slots to spawn into
        dormant_slots = torch.nonzero(self.alive == 0).flatten()
        
        for parent_idx in parents:
            if len(dormant_slots) > 0:
                child_idx = dormant_slots[0] # Take first available slot
                dormant_slots = dormant_slots[1:] # Remove from available list
                
                # Mitosis Logic
                parent_energy = self.F[parent_idx].item()
                energy_transfer = parent_parent_energy = parent_energy * 0.5
                
                # Update Parent
                self.F[parent_idx] = energy_transfer
                
                # Initialize Child
                self.activate_node(
                    child_idx, 
                    energy=energy_transfer, 
                    conductivity=self.sigma[parent_idx].item() # Inherit genetics
                )
                
                # Connect Child to Parent (Grow Branch)
                self.adj_energy[parent_idx, child_idx] = 1.0
                self.adj_energy[child_idx, parent_idx] = 1.0
                self.adj_info[parent_idx, child_idx] = 1.0
                self.adj_info[child_idx, parent_idx] = 1.0
                
                births += 1
            else:
                break # No space left in the universe

        return births, deaths

    # --- Standard Physics Steps (With Alive Mask applied) ---
    def run_signaling_phase(self):
        stress_condition = (self.F > 3.0).float() * self.alive # Only living scream
        self.scream_counter = (self.scream_counter + 1) * stress_condition
        voice_stamina = (self.scream_counter <= 30).float()
        original_scream = stress_condition * voice_stamina
        
        # Repeater
        is_root = (self.a > 0).float()
        hearing_whisper = ((self.I > 0.005) & (self.I < 0.1)).float()
        repeater_boost = is_root * hearing_whisper * 0.5
        total_source = torch.max(original_scream, repeater_boost)
        
        # Propagate (Masked by Topology)
        diff_matrix = self.I.unsqueeze(1) - self.I.unsqueeze(0)
        S_flow = F.relu(diff_matrix) * self.adj_info 
        S_in = S_flow.sum(dim=0)
        S_out = S_flow.sum(dim=1)
        
        self.I = self.I - (S_out * 0.2) + (S_in * 0.2) + total_source
        self.I = self.I * 0.98 * self.alive # Dead nodes carry no signal

    def update_strategy(self):
        self.danger_mask = (self.F > 3.0).float()
        warning_mask = (self.I > 0.1).float()
        lockdown_mask = torch.clamp(self.danger_mask + warning_mask, max=1.0)
        self.g_lateral = (1.0 - lockdown_mask) * self.alive # Dead nodes are closed

    def step(self, t):
        # 1. Structural Evolution (Slow timescale check, e.g., every tick or every 10)
        b, d = self.run_evolutionary_dynamics()
        
        # 2. Fast Physics
        self.run_signaling_phase()
        self.update_strategy()

        diff_matrix = self.F.unsqueeze(1) - self.F.unsqueeze(0)
        P_matrix = F.relu(diff_matrix)
        effective_adj = self.adj_energy * self.g_lateral.unsqueeze(0) * self.g_lateral.unsqueeze(1)
        J = self.alpha_E * P_matrix * effective_adj
        
        G_out = J.sum(dim=1) * self.dt 
        G_in = J.sum(dim=0) * self.dt

        safe_factor = torch.clamp(self.a * self.sigma * self.dt, max=0.9)
        potential_diff = F.relu(self.phi_res - self.F)
        G_res = safe_factor * potential_diff * self.alive # Only living drink

        # Update F
        F_new = (self.F - G_out + (G_in * (1.0 - self.gamma)) + G_res) * (1.0 - self.metabolic_rate)
        
        # Noise
        if self.noise_level > 0:
            noise = torch.randn_like(F_new) * self.noise_level
            F_new = F_new + noise
            
        self.F = F.relu(F_new) * self.alive # Apply Death Mask
        
        # Conductivity Adaptation
        if G_out.sum() > 0:
            epsilon = (G_in) / (G_out + 1e-6)
            delta_sigma = self.eta_sigma * (torch.log1p(epsilon) - 0.2)
            self.sigma = F.relu(self.sigma + delta_sigma)

        self.telemetry = {
            "tick": t,
            "alive_count": self.alive.sum().item(),
            "total_energy": self.F.sum().item(),
            "births": b,
            "deaths": d
        }
        return self.F.numpy()