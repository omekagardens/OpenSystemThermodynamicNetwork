import torch
import torch.nn.functional as F
import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional

@dataclass
class DETConfig:
    """Physics Constants & Hyperparameters"""
    phi_res: float = 1.0        # Reservoir Potential
    dt: float = 0.01            # Time step
    noise_level: float = 0.01   # Thermal Noise (0.0 to 1.0)
    gamma: float = 0.02         # Entropy/Friction loss
    eta_sigma: float = 0.1      # Learning rate (plasticity)
    alpha_E: float = 1.0        # Energy Flow Rate
    metabolic_rate: float = 0.01 # Maintenance cost
    decay_I: float = 0.98       # Signal decay rate
    
    # Thresholds
    threshold_stress: float = 3.0
    threshold_birth: float = 5.0
    threshold_death: float = 0.001

class DETSystem:
    def __init__(self, num_nodes: int, config: DETConfig):
        self.N = num_nodes
        self.cfg = config
        
        # --- State Vector ---
        self.F = torch.zeros(num_nodes)
        self.sigma = torch.ones(num_nodes) * 0.5
        self.a = torch.zeros(num_nodes)      # Soil Access
        self.I = torch.zeros(num_nodes)      # Signal
        self.alive = torch.zeros(num_nodes)  # 0=Dead, 1=Alive
        
        # --- Control Vector ---
        self.g_lateral = torch.ones(num_nodes)
        self.scream_counter = torch.zeros(num_nodes)
        
        # --- Topology ---
        self.adj_energy = torch.zeros(num_nodes, num_nodes)
        self.adj_info = torch.zeros(num_nodes, num_nodes)
        
        # --- Hooks (API Extension) ---
        # Allow users to inject logic: self.hooks['pre_step'].append(my_func)
        self.hooks: Dict[str, List[Callable]] = {
            'pre_step': [],
            'post_step': []
        }
        
        self.tick = 0
        self.telemetry = {}

    def register_hook(self, trigger: str, func: Callable):
        """Expand functionality without subclassing."""
        if trigger in self.hooks:
            self.hooks[trigger].append(func)

    def activate_node(self, idx, energy=1.0):
        self.alive[idx] = 1.0
        self.F[idx] = energy
        self.sigma[idx] = 0.5
        self.g_lateral[idx] = 1.0

    def step(self):
        # Ensure topology/control arrays are torch tensors (some builders may set numpy arrays)
        a = self.a
        if not torch.is_tensor(a):
            a = torch.as_tensor(a, dtype=self.F.dtype, device=self.F.device)

        # 1. Run User Hooks (Pre-Integration)
        for func in self.hooks['pre_step']:
            func(self)

        # 2. Signaling Phase (Information Layer)
        stress = (self.F > self.cfg.threshold_stress).float() * self.alive
        self.scream_counter = (self.scream_counter + 1) * stress
        # Voice Stamina: Can only scream for 30 ticks
        voice = (self.scream_counter <= 30).float()
        source = stress * voice
        
        # Repeater Logic (Roots Boost Signals)
        # IMPORTANT: Gate the repeater by the lateral gate. If the plant is already "closed",
        # it should not keep re-amplifying the warning forever (prevents high-lock saturation).
        is_root = (a > 0).float()
        whisper = ((self.I > 0.005) & (self.I < 0.1)).float()
        source = torch.max(source, is_root * whisper * 0.5 * self.g_lateral)
        
        # Propagate Info
        dI = self.I.unsqueeze(1) - self.I.unsqueeze(0)
        flow_I = F.relu(dI) * self.adj_info
        self.I = self.I - (flow_I.sum(1)*0.2) + (flow_I.sum(0)*0.2) + source
        self.I = self.I * self.cfg.decay_I * self.alive
        # Keep the information channel as a nonnegative "amplitude" signal.
        # (Prevents oscillatory negative values from contaminating warning logic and logging.)
        self.I = F.relu(self.I)
        
        # 3. Strategy Phase (Gating)
        # Lock if Hurt (>3.0) OR Warned (>0.1)
        danger = (self.F > self.cfg.threshold_stress).float()
        warn = (self.I > 0.1).float()
        lock = torch.clamp(danger + warn, max=1.0)

        # Gate target: close when locked, open when safe. Relax toward target to allow recovery (avoid saturation).
        target_gate = (1.0 - lock) * self.alive
        k_recover = 0.15  # 0..1 : higher = faster open/close; lower = smoother recovery
        self.g_lateral = torch.clamp(self.g_lateral + k_recover * (target_gate - self.g_lateral), 0.0, 1.0)

        # 4. Metabolic Phase (Energy Layer)
        dF = self.F.unsqueeze(1) - self.F.unsqueeze(0)
        flow_E = self.cfg.alpha_E * F.relu(dF) * self.adj_energy * \
                 self.g_lateral.unsqueeze(0) * self.g_lateral.unsqueeze(1)
        
        G_out = flow_E.sum(1) * self.cfg.dt
        G_in = flow_E.sum(0) * self.cfg.dt
        
        # Reservoir Inflow
        safe_sigma = torch.clamp(a * self.sigma * self.cfg.dt, max=0.9)
        G_res = safe_sigma * F.relu(self.cfg.phi_res - self.F) * self.alive
        
        # Update F (Conservation + Metabolism)
        F_new = (self.F - G_out + (G_in * (1.0 - self.cfg.gamma)) + G_res) * (1.0 - self.cfg.metabolic_rate)
        
        # Thermal Noise
        if self.cfg.noise_level > 0:
            noise = torch.randn_like(F_new) * self.cfg.noise_level
            F_new += noise
            
        self.F = F.relu(F_new) * self.alive
        
        # 5. Plasticity (Learning)
        if G_out.sum() > 0:
            eff = (G_in) / (G_out + 1e-6)
            ds = self.cfg.eta_sigma * (torch.log1p(eff) - 0.2)
            self.sigma = F.relu(self.sigma + ds)

        # 6. Run User Hooks (Post-Integration)
        for func in self.hooks['post_step']:
            func(self)
            
        self.tick += 1
        
        # Standard Telemetry
        self.telemetry = {
            "tick": self.tick,
            "energy": self.F.sum().item(),
            "locked": (self.g_lateral < 0.5).sum().item(),
            "alive": self.alive.sum().item()
        }
        return self.telemetry

    def save(self, filename="det_snapshot.pkl"):
        """Save simulation state to file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    @staticmethod
    def load(filename="det_snapshot.pkl"):
        """Load simulation state."""
        with open(filename, 'rb') as f:
            return pickle.load(f)