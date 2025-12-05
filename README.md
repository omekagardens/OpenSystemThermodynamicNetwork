# OpenSystemThermodynamicNetwork
The universe is modeled as a directed graph coupled to an external infinite reservoir.

### **1. Nomenclature Mapping**

| Previous Term | Rigorous Physical/Mathematical Label | Symbol |
| :--- | :--- | :--- |
| **Active Negentropy Export** (Directed Flux) | $J_{i \to j}$ |
| **Reservoir Influx** | $J_{res \to i}$ |
| **Infinite Potential Bath** (Thermodynamic Reservoir) | $\Phi_{res}$ |
| **Gradient-Driven Coupling** | $\nabla \Phi$ |
| **High-Conductivity Emitter** | $\sigma_{high}$ |
| **Local Accumulator** | $\sigma_{low}$ |
| **Flux Allocation Strategy** | $S(F_i)$ |

---

### **2. System Definition: Open-System Thermodynamic Network**

The universe is modeled as a directed graph coupled to an external infinite reservoir.

**A. The Resource ($\Psi$)**
A composite scalar potential representing the capacity for work.
$$\Psi = \alpha_E E + \alpha_I I + \alpha_T T$$
*(Energy, Information, Temporal Availability)*

**B. The Conservation Law (with Source terms)**
For any node $i$, the rate of change of local potential $F_i$ is:

$$\frac{dF_i}{dt} = \underbrace{J_{res \to i}}_{\text{Reservoir Influx}} + \underbrace{\sum_{k} \eta J_{k \to i}}_{\text{Network Inflow}} - \underbrace{\gamma \sum_{j} J_{i \to j}}_{\text{Active Export}}$$

* **$\gamma > 1$ (Dissipation Coefficient):** Represents the thermodynamic cost of generating ordered flux (internal entropy production).
* **$\eta < 1$ (Coupling Efficiency):** Represents transmission losses between nodes.

---

### **3. The Two Dynamic Regimes**

We analyzed two distinct boundary conditions for the Reservoir Influx ($J_{res \to i}$).

#### **Regime A: Static Background Field (formerly "Rain")**
The reservoir provides a constant flux independent of the node's state.
$$J_{res \to i} = C$$
* **Result:** **Accumulators** (Hoarders) dominate. They capture the constant influx and minimize export costs ($\gamma$). **Emitters** deplete rapidly as export costs exceed static income. System entropy is high due to stagnation.

#### **Regime B: Potential-Dependent Coupling (formerly "Responsive/Kenotic")**
The reservoir flux is driven by the potential gradient (difference) between the Reservoir ($\Phi_{res}$) and the Node ($F_i$). This follows standard diffusion/osmosis laws.
$$J_{res \to i} = \sigma \cdot \max(0, \Phi_{res} - F_i)$$
* **$\sigma$:** Conductivity coefficient.
* **$\Phi_{res} - F_i$:** The Potential Gradient.

**Physical Consequence:**
1.  **Accumulators** saturate their local potential ($F_i \to \Phi_{res}$). The gradient vanishes ($\Delta \Phi \to 0$), and reservoir influx stops ($J_{res \to i} \to 0$). They become thermodynamically isolated.
2.  **Emitters** continuously export potential to the network ($J_{i \to j} > 0$). This maintains a low local potential ($F_i \ll \Phi_{res}$), sustaining a steep gradient.
3.  **Result:** Emitters act as **Super-Conductors**, maximizing the total power throughput of the system. They draw continuous massive power from the reservoir and distribute it, driving the network far from equilibrium (life-like behavior).

---
