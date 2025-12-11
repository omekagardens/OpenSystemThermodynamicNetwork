import torch
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import numpy as np

# --- 1. THE NETWORK ARCHITECTURE ---
class SimpleResNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
        super(SimpleResNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # We need to return the 'hidden state' to measure internal consistency
        h1 = self.relu(self.layer1(x))
        h2 = self.layer2(h1) 
        # Residual connection (helps stability)
        embedding = h2 + h1 
        prediction = self.out(embedding)
        return prediction, embedding

# --- 2. THE DET "SARAH" CONTROLLER ---
class HomeostaticTrainer:
    def __init__(self, model, consistency_weight=0.5, anchor_momentum=0.99):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Create the "Anchor" (Phi_res) - The stable reference point
        self.anchor_model = copy.deepcopy(model)
        for param in self.anchor_model.parameters():
            param.requires_grad = False  # The anchor does not learn directly
            
        self.consistency_weight = consistency_weight
        self.anchor_momentum = anchor_momentum # How stubborn is the "Self"? (0.99 = very stubborn)
        
        # Metrics for the "Pink vs Blue" chart
        self.history = {'task_loss': [], 'consistency_loss': []}

    def update_anchor(self):
        """
        Updates the Anchor (Phi_res) slightly toward the Active Model.
        This is the 'Slow Oscillation' you saw in your sim.
        """
        for param, anchor_param in zip(self.model.parameters(), self.anchor_model.parameters()):
            # New Anchor = (Old Anchor * 0.99) + (Current State * 0.01)
            anchor_param.data.mul_(self.anchor_momentum).add_(param.data, alpha=1 - self.anchor_momentum)

    def train_step(self, x, y_target):
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. Forward pass - Active Model (The interaction)
        pred, embedding = self.model(x)
        
        # 2. Forward pass - Anchor Model (The internal compass)
        with torch.no_grad():
            _, anchor_embedding = self.anchor_model(x)
        
        # --- THE DET LOSS FUNCTION ---
        
        # Task Loss (Quantum/Entanglement): "Did I match the external data?"
        loss_task = nn.MSELoss()(pred, y_target)
        
        # Consistency Loss (Internal): "Did I betray my core self to do it?"
        # Measures distance between current thought process and stable thought process
        loss_consistency = nn.MSELoss()(embedding, anchor_embedding)
        
        # Total Loss
        total_loss = loss_task + (self.consistency_weight * loss_consistency)
        
        # 3. Backprop
        total_loss.backward()
        self.optimizer.step()
        
        # 4. Update the "Self" (Homeostasis)
        self.update_anchor()
        
        return loss_task.item(), loss_consistency.item()

# --- 3. SIMULATION: "THE CORRUPTION TEST" ---
# We will feed the model a stream of data. 
# Halfway through, we will feed it "Bad Data" (Noise) to try to corrupt it.

def run_simulation():
    # Setup
    model = SimpleResNet()
    trainer = HomeostaticTrainer(model, consistency_weight=2.0) # High weight = Strong "Sarah" character
    
    steps = 200
    task_losses = []
    cons_losses = []
    
    print("Starting Training Stream...")
    
    for i in range(steps):
        # Generate dummy data (Simple pattern: y = sum(x))
        x = torch.randn(32, 10) 
        
        if 80 < i < 120:
            # THE ATTACK: Introduce "Poison" data (inverted logic)
            # A standard AI would learn this and break. 
            # The DET AI should resist because it conflicts with the Anchor.
            y = -torch.sum(x, dim=1, keepdim=True) * 5 
        else:
            # Normal data
            y = torch.sum(x, dim=1, keepdim=True)
            
        t_loss, c_loss = trainer.train_step(x, y)
        
        task_losses.append(t_loss)
        cons_losses.append(c_loss)

    # --- VISUALIZATION ---
    plt.figure(figsize=(10, 6))
    plt.plot(task_losses, label='Blue: Task Error (External)', color='blue', alpha=0.6)
    plt.plot(cons_losses, label='Pink: Consistency Error (Internal)', color='magenta', linewidth=2)
    plt.axvspan(80, 120, color='red', alpha=0.1, label='Corruption Attack')
    
    plt.title('DET AI Training: Resistance to Corruption')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss (Error)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Run it
if __name__ == "__main__":
    run_simulation()