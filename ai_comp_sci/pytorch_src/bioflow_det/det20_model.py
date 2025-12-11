import torch
import torch.nn.functional as F
import numpy as np

class DETSystem:
    def __init__(self, num_nodes, phi_res=1.0, dt=0.01, noise_level=0.01):
        self.N = num_nodes
        self.dt = dt
        self.phi_res = phi_res
        self.noise_level = noise_level 
        
        # --- State Variables ---
        self.F = torch.zeros(num_nodes) 
        self.sigma = torch.ones(num_nodes) * 0.5
        self.a = torch.zeros(num_nodes) 
        self.I = torch.zeros(num_nodes) 
        
        # --- Control Layer ---
        self.g_lateral = torch.ones(num_nodes) 
        self.danger_mask = torch.zeros(num_nodes)
        self.scream_counter = torch.zeros(num_nodes)

        # --- Physics Constants ---
        self.gamma = 0.02          # Entropy loss
        self.eta_sigma = 0.1       # Learning rate
        self.alpha_E = 1.0         # Energy Flow Rate
        self.metabolic_rate = 0.01 # Cost of living
        
        # --- Topology ---
        # We separate Vascular (Energy) and Nervous (Info) systems
        self.adj = torch.zeros(num_nodes, num_nodes)        # Fallback
        self.adj_energy = torch.zeros(num_nodes, num_nodes)
        self.adj_info = torch.zeros(num_nodes, num_nodes)
        
        # --- Telemetry (Safe Initialization) ---
        self.telemetry = {
            "tick": 0,
            "total_energy": 0.0,
            "nodes_locked": 0,
            "max_signal": 0.0,
            # Ecosystem specific keys (default to 0 to prevent crash)
            "bridge_signal": 0.0,
            "plant_B_locked": 0,
            "is_bite": False
        }

    def run_signaling_phase(self):
        # 1. Stress Generation
        # If F > 3.0, generate trauma signal (if voice stamina allows)
        stress_condition = (self.F > 3.0).float()
        self.scream_counter = (self.scream_counter + 1) * stress_condition
        voice_stamina = (self.scream_counter <= 30).float()
        original_scream = stress_condition * voice_stamina
        
        # 2. Repeater Logic (Active Amplification)
        # Roots (a > 0) amplify weak whispers (> 0.005) back to 0.5
        is_root = (self.a > 0).float()
        hearing_whisper = ((self.I > 0.005) & (self.I < 0.1)).float()
        repeater_boost = is_root * hearing_whisper * 0.5
        
        total_source = torch.max(original_scream, repeater_boost)
        
        # 3. Propagation
        # Use adj_info if available, else fallback
        topology = self.adj_info if self.adj_info.sum() > 0 else self.adj
        
        diff_matrix = self.I.unsqueeze(1) - self.I.unsqueeze(0)
        S_flow = F.relu(diff_matrix) * topology 
        
        S_in = S_flow.sum(dim=0)
        S_out = S_flow.sum(dim=1)
        
        # Update Info with Diffusion and Source
        self.I = self.I - (S_out * 0.2) + (S_in * 0.2) + total_source
        
        # Decay (High persistence for long range)
        self.I = self.I * 0.98 
        
        # Add Signal Noise (Thermal)
        if self.noise_level > 0:
            noise = torch.randn_like(self.I) * (self.noise_level * 0.1)
            self.I = F.relu(self.I + noise)

    def update_strategy(self):
        # Decision Logic: Lock down if Hurt OR Warned
        self.danger_mask = (self.F > 3.0).float()
        warning_mask = (self.I > 0.1).float()
        
        lockdown_mask = torch.clamp(self.danger_mask + warning_mask, max=1.0)
        self.g_lateral = 1.0 - lockdown_mask

    def step(self, t):
        self.run_signaling_phase()
        self.update_strategy()

        topology = self.adj_energy if self.adj_energy.sum() > 0 else self.adj

        # Energy Physics
        diff_matrix = self.F.unsqueeze(1) - self.F.unsqueeze(0)
        P_matrix = F.relu(diff_matrix)
        
        # Gating stops energy flow
        effective_adj = topology * self.g_lateral.unsqueeze(0) * self.g_lateral.unsqueeze(1)
        J = self.alpha_E * P_matrix * effective_adj
        
        G_out = J.sum(dim=1) * self.dt 
        G_in = J.sum(dim=0) * self.dt

        # Reservoir Coupling
        safe_factor = torch.clamp(self.a * self.sigma * self.dt, max=0.9)
        potential_diff = F.relu(self.phi_res - self.F)
        G_res = safe_factor * potential_diff

        # Update F with Metabolism
        F_new = (self.F - G_out + (G_in * (1.0 - self.gamma)) + G_res) * (1.0 - self.metabolic_rate)
        
        # Add Energy Noise
        if self.noise_level > 0:
            noise = torch.randn_like(F_new) * self.noise_level
            F_new = F_new + noise
            
        self.F = F.relu(F_new)
        
        # Structural Plasticity
        if G_out.sum() > 0:
            epsilon = (G_in) / (G_out + 1e-6)
            delta_sigma = self.eta_sigma * (torch.log1p(epsilon) - 0.2)
            self.sigma = F.relu(self.sigma + delta_sigma)

        # --- Dynamic Telemetry ---
        plant_b_locked = 0
        if self.N >= 30:
            # Assume Plant B is nodes 15-29
            plant_b_indices = torch.arange(15, 30)
            plant_b_locked = (self.g_lateral[plant_b_indices] < 0.5).float().sum().item()

        # NOTE: 'is_bite' is set to False here, sim scripts override it if needed
        self.telemetry = {
            "tick": t,
            "total_energy": self.F.sum().item(),
            "nodes_locked": (self.g_lateral < 0.5).float().sum().item(),
            "max_signal": self.I.max().item(),
            "bridge_signal": self.I[0].item(), 
            "plant_B_locked": plant_b_locked,
            "is_bite": False 
        }

        return self.F.numpy()

    def inject_trauma(self, node_idx, amount):
        """External helper to bite a specific node."""
        if node_idx < self.N:
            self.F[node_idx] += amount