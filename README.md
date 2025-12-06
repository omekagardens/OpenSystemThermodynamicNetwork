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

## Simulator Ready Model

1. Entities and Topology
	•	Let $\mathcal{A} = \{1, 2, \dots, N\}$ be the set of nodes.
	•	Let 0 denote a distinguished reservoir node.
	•	Interactions occur on a directed graph over nodes \{0\} \cup \mathcal{A}.

⸻

2. State Variables

For each node i \in \mathcal{A}:
	•	F_i(t) \in \mathbb{R}: scalar state (e.g. “free-level” or potential-like quantity).
	•	\sigma_i(t) \ge 0: conductivity with respect to the reservoir.
	•	a_i(t) \in [0,1]: gating factor for reservoir coupling.

Global constant:
	•	\Phi_{\text{res}} \in \mathbb{R}: reservoir potential (assumed large and approximately constant).

⸻

3. Resource Flows Between Nodes

Define a composite flow from node i to node j:

J_{i \to j}(t) = \alpha_E P_{i \to j}(t)
+ \alpha_I \dot{I}_{i \to j}(t)
+ \alpha_T A_{i \to j}(t)

where:
	•	P_{i \to j}(t): physical power from i to j (J/s),
	•	\dot{I}_{i \to j}(t): information rate (bits/s),
	•	A_{i \to j}(t): dimensionless activity/attention rate,
	•	\alpha_E, \alpha_I, \alpha_T \ge 0: fixed weighting coefficients.

Over a discrete time interval (tick) [t_k, t_{k+1}], define:

G_{i \to j}^{(k)}
= \int_{t_k}^{t_{k+1}} J_{i \to j}(t)\, dt

Total outgoing and incoming flows for node i (excluding reservoir):

G_i^{\text{out}, (k)}
= \sum_{j \in \mathcal{A}} G_{i \to j}^{(k)}

R_i^{(k)}
= \sum_{j \in \mathcal{A}} G_{j \to i}^{(k)}

⸻

4. Potential-Dependent Coupling with Reservoir

Reservoir–node coupling flux (continuous form):

J_{\text{res} \to i}(t)
= a_i(t)\, \sigma_i(t)\, \max\big(0,\; \Phi_{\text{res}} - F_i(t)\big)

Discrete tick approximation with \Delta t = t_{k+1} - t_k:

G_i^{\text{res}, (k)}
= J_{\text{res} \to i}^{(k)} \, \Delta t
= a_i^{(k)}\, \sigma_i^{(k)}\, \max\big(0,\; \Phi_{\text{res}} - F_i^{(k)}\big)\, \Delta t

Total incoming flow for node i including reservoir:

R_i^{\text{tot}, (k)}
= R_i^{(k)} + G_i^{\text{res}, (k)}

⸻

5. Free-Level Update per Tick

Let \gamma > 0 be a cost coefficient and \eta_{j \to i} \in [0,1] be transfer-efficiency factors for flows from j to i.

A simple linear update rule for F_i over one tick:

F_i^{(k+1)}
= F_i^{(k)}
- \gamma\, G_i^{\text{out}, (k)}
+ \sum_{j \in \mathcal{A}} \eta_{j \to i}\, G_{j \to i}^{(k)}
+ G_i^{\text{res}, (k)}

Interpretation:
	•	First term: persistence of previous level.
	•	Second term: reduction due to outgoing flow.
	•	Third term: increase due to incoming flows from other nodes.
	•	Fourth term: increase due to reservoir coupling driven by potential difference.

⸻

6. Adaptation of Conductivity (Optional)

Define a per-tick efficiency measure, e.g.:

\epsilon_i^{(k)}
= \frac{R_i^{\text{tot}, (k)}}{G_i^{\text{out}, (k)} + \varepsilon}

with small \varepsilon > 0 to avoid division by zero.

Let f: \mathbb{R} \to \mathbb{R} be a bounded update function (e.g. sigmoid-like). Then:

\sigma_i^{(k+1)}
= \sigma_i^{(k)} + \eta_\sigma\, f\big(\epsilon_i^{(k)}\big)

where \eta_\sigma > 0 is a small learning rate.

⸻

7. Summary of Core Equations
	1.	Inter-node flow per tick
G_{i \to j}^{(k)} = \int_{t_k}^{t_{k+1}}
\big(\alpha_E P_{i \to j}(t)
+ \alpha_I \dot{I}_{i \to j}(t)
+ \alpha_T A_{i \to j}(t)\big)\, dt
	2.	Reservoir coupling per tick
G_i^{\text{res}, (k)}
= a_i^{(k)}\, \sigma_i^{(k)}\, \max\big(0,\; \Phi_{\text{res}} - F_i^{(k)}\big)\, \Delta t
	3.	Free-level update
F_i^{(k+1)}
= F_i^{(k)}
- \gamma\, G_i^{\text{out}, (k)}
+ \sum_{j \in \mathcal{A}} \eta_{j \to i}\, G_{j \to i}^{(k)}
+ G_i^{\text{res}, (k)}
	4.	Conductivity adaptation (optional)
\epsilon_i^{(k)}
= \frac{R_i^{\text{tot}, (k)}}{G_i^{\text{out}, (k)} + \varepsilon},
\quad
\sigma_i^{(k+1)}
= \sigma_i^{(k)} + \eta_\sigma\, f\big(\epsilon_i^{(k)}\big)

This card is self-contained and ready to drop into a simulator without any explicit semantic labels attached.
