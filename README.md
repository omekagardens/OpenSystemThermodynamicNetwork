# OpenSystemThermodynamicNetwork (Previously DET)
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

Simulator Ready Model (DET 2.0)

1. Entities and Topology
	•	Let $\mathcal{A} = {1, 2, \dots, N}$ be the set of nodes.
	•	Let $0$ denote a distinguished reservoir node.
	•	Interactions occur on a directed graph over nodes ${0} \cup \mathcal{A}$.

⸻

2. State Variables

For each node $i \in \mathcal{A}$:
	•	$F_i(t) \in \mathbb{R}$
	•	$\sigma_i(t) \ge 0$
	•	$a_i(t) \in [0,1]$

Global constant:
	•	$\Phi_{\text{res}} \in \mathbb{R}$

⸻

3. Resource Flows Between Nodes

Composite flow:

$$
J_{i \to j}(t)
= \alpha_E P_{i \to j}(t)
	•	\alpha_I \dot{I}_{i \to j}(t)
	•	\alpha_T A_{i \to j}(t)
$$

Discrete flow:

$$
G_{i \to j}^{(k)}
= \int_{t_k}^{t_{k+1}} J_{i \to j}(t), dt
$$

Outgoing:

$$
G_i^{\text{out},(k)}
= \sum_{j \in \mathcal{A}} G_{i \to j}^{(k)}
$$

Incoming:

$$
R_i^{(k)}
= \sum_{j \in \mathcal{A}} G_{j \to i}^{(k)}
$$

⸻

4. Potential-Dependent Coupling With Reservoir

Continuous:

$$
J_{\text{res} \to i}(t)
= a_i(t), \sigma_i(t), \max(0,; \Phi_{\text{res}} - F_i(t))
$$

Discrete:

$$
G_i^{\text{res},(k)}
= a_i^{(k)}, \sigma_i^{(k)}, \max(0,; \Phi_{\text{res}} - F_i^{(k)}), \Delta t
$$

Total incoming:

$$
R_i^{\text{tot},(k)}
= R_i^{(k)} + G_i^{\text{res},(k)}
$$

⸻

5. Free-Level Update per Tick

$$
F_i^{(k+1)}
= F_i^{(k)}
	•	\gamma, G_i^{\text{out},(k)}

	•	\sum_{j \in \mathcal{A}} \eta_{j \to i}, G_{j \to i}^{(k)}
	•	G_i^{\text{res},(k)}
$$

⸻

6. Adaptation of Conductivity (Optional)

Efficiency:

$$
\epsilon_i^{(k)}
= \frac{R_i^{\text{tot},(k)}}{G_i^{\text{out},(k)} + \varepsilon}
$$

Update:

$$
\sigma_i^{(k+1)}
= \sigma_i^{(k)} + \eta_\sigma, f(\epsilon_i^{(k)})
$$

⸻

7. Summary of Core Equations

1. Inter-node flow

$$
G_{i \to j}^{(k)}
= \int_{t_k}^{t_{k+1}}
\left(
\alpha_E P_{i \to j}(t)
	•	\alpha_I \dot{I}_{i \to j}(t)
	•	\alpha_T A_{i \to j}(t)
\right) dt
$$

2. Reservoir coupling

$$
G_i^{\text{res},(k)}
= a_i^{(k)}, \sigma_i^{(k)},
\max(0,; \Phi_{\text{res}} - F_i^{(k)}), \Delta t
$$

3. Free-level update

$$
F_i^{(k+1)}
= F_i^{(k)}
	•	\gamma, G_i^{\text{out},(k)}

	•	\sum_{j \in \mathcal{A}} \eta_{j \to i}, G_{j \to i}^{(k)}
	•	G_i^{\text{res},(k)}
$$

4. Conductivity update

$$
\epsilon_i^{(k)}
= \frac{R_i^{\text{tot},(k)}}{G_i^{\text{out},(k)} + \varepsilon}
$$

$$
\sigma_i^{(k+1)}
= \sigma_i^{(k)} + \eta_\sigma, f(\epsilon_i^{(k)})
$$
