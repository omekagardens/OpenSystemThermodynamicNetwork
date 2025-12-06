# OpenSystemThermodynamicNetwork (Previously DET)

The universe is modeled as a directed graph coupled to an external infinite reservoir.

---

## **1. Nomenclature Mapping**

| Previous Term | Rigorous Physical/Mathematical Label | Symbol |
| :--- | :--- | :--- |
| Active Negentropy Export | Directed Flux | $J_{i \to j}$ |
| Reservoir Influx | $J_{\text{res} \to i}$ |
| Infinite Potential Bath | Reservoir Potential | $\Phi_{\text{res}}$ |
| Gradient-Driven Coupling | Potential Gradient | $\nabla \Phi$ |
| High-Conductivity Emitter | High Conductivity | $\sigma_{\text{high}}$ |
| Local Accumulator | Low Conductivity | $\sigma_{\text{low}}$ |
| Flux Allocation Strategy | Allocation Function | $S(F_i)$ |

---

## **2. System Definition: Open-System Thermodynamic Network**

### **A. Composite Resource Potential**

$$
\Psi = \alpha_E E + \alpha_I I + \alpha_T T
$$

### **B. Conservation Law with Sources**

$$
\frac{dF_i}{dt}
= J_{\text{res} \to i}
+ \sum_k \eta\, J_{k \to i}
- \gamma \sum_j J_{i \to j}
$$

---

## **3. Dynamic Regimes**

### **Regime A: Static Reservoir Flux**

$$
J_{\text{res} \to i} = C
$$

### **Regime B: Potential‑Dependent Coupling**

$$
J_{\text{res} \to i}
= \sigma_i \max(0,\, \Phi_{\text{res}} - F_i)
$$

---

# **Simulator Ready Model (DET 2.0)**

## **1. Entities and Topology**

- Let $\mathcal{A} = \{1,2,\dots,N\}$ be the active nodes.  
- Let $0$ denote the reservoir node.  
- Interactions occur on a directed graph over $\{0\} \cup \mathcal{A}$.

---

## **2. State Variables**

For each node $i \in \mathcal{A}$:

- $F_i(t) \in \mathbb{R}$  
- $\sigma_i(t) \ge 0$  
- $a_i(t) \in [0,1]$

Global constant:

- $\Phi_{\text{res}} \in \mathbb{R}$

---

## **3. Resource Flows Between Nodes**

### **Composite Flow**

$$
J_{i \to j}(t)
= \alpha_E P_{i \to j}(t)
+ \alpha_I \dot{I}_{i \to j}(t)
+ \alpha_T A_{i \to j}(t)
$$

### **Discrete Tick Flow**

$$
G_{i \to j}^{(k)}
= \int_{t_k}^{t_{k+1}} J_{i \to j}(t)\, dt
$$

Outgoing:

$$
G_i^{\text{out},(k)} = \sum_{j \in \mathcal{A}} G_{i \to j}^{(k)}
$$

Incoming:

$$
R_i^{(k)} = \sum_{j \in \mathcal{A}} G_{j \to i}^{(k)}
$$

---

## **4. Potential‑Dependent Reservoir Coupling**

### **Continuous**

$$
J_{\text{res} \to i}(t)
= a_i(t)\, \sigma_i(t)\, \max(0,\, \Phi_{\text{res}} - F_i(t))
$$

### **Discrete**

$$
G_i^{\text{res},(k)}
= a_i^{(k)}\, \sigma_i^{(k)}\, \max(0,\, \Phi_{\text{res}} - F_i^{(k)})\, \Delta t
$$

Total incoming:

$$
R_i^{\text{tot},(k)}
= R_i^{(k)} + G_i^{\text{res},(k)}
$$

---

## **5. Free‑Level Update per Tick**

$$
F_i^{(k+1)}
= F_i^{(k)}
- \gamma\, G_i^{\text{out},(k)}
+ \sum_{j \in \mathcal{A}} \eta_{j \to i}\, G_{j \to i}^{(k)}
+ G_i^{\text{res},(k)}
$$

---

## **6. Conductivity Adaptation (Optional)**

Efficiency:

$$
\epsilon_i^{(k)}
= \frac{R_i^{\text{tot},(k)}}{G_i^{\text{out},(k)} + \varepsilon}
$$

Update:

$$
\sigma_i^{(k+1)}
= \sigma_i^{(k)} + \eta_\sigma\, f(\epsilon_i^{(k)})
$$

---

## **7. Summary of Core Equations**

### **(1) Inter-Node Flow**

$$
G_{i \to j}^{(k)}
= \int_{t_k}^{t_{k+1}}
\left(
\alpha_E P_{i \to j}(t)
+ \alpha_I \dot{I}_{i \to j}(t)
+ \alpha_T A_{i \to j}(t)
\right)\, dt
$$

### **(2) Reservoir Coupling**

$$
G_i^{\text{res},(k)}
= a_i^{(k)}\, \sigma_i^{(k)}\, \max\!\bigl(0,\, \Phi_{\text{res}} - F_i^{(k)}\bigr)\, \Delta t
$$

### **(3) Free-Level Update**

$$
F_i^{(k+1)}
= F_i^{(k)}
- \gamma\, G_i^{\text{out},(k)}
+ \sum_{j \in \mathcal{A}} \eta_{j \to i}\, G_{j \to i}^{(k)}
+ G_i^{\text{res},(k)}
$$

### **(4) Conductivity Update**

$$
\epsilon_i^{(k)}
= \frac{R_i^{\text{tot},(k)}}{G_i^{\text{out},(k)} + \varepsilon}
$$

$$
\sigma_i^{(k+1)}
= \sigma_i^{(k)} + \eta_\sigma\, f\!\bigl(\epsilon_i^{(k)}\bigr)
$$
