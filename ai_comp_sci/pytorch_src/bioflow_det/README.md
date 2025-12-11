
### A Dimensionally Consistent Framework for Distributed Bio-Intelligence

**DET 2.0** is a minimal, physically consistent dynamical systems model that unifies **Thermodynamic Resource Flow** with **Active Topological Control**.

Unlike traditional biological models (which often ignore the energy cost of computation) or neural networks (which are abstract mathematical constructs), DET 2.0 models "Life" as a contest between two flows:

1.  **Energy ($P$):** A conserved quantity that builds structure but moves slowly.
2.  **Information ($\dot{I}$):** A control signal that moves quickly to modify the structure before the energy arrives.

This repository contains the Python implementation of the theory, demonstrating emergent behaviors such as **Systemic Acquired Resistance (SAR)**, **Antifragility**, and **Inter-Species Warning Networks**.

-----

## Core Features

  * **Dimensional Discipline:** The entire physics engine is derived from only two primitive dimensions: **Resource Quantity $[Q]$** and **Time $[T]$**.
  * **Composite Flow:** Modeling of dual-layer topology—**Vascular Networks** (Phloem/Xylem) for energy and **Signaling Networks** (Nerves/Mycelium) for information.
  * **The "Paul Revere" Protocol:** An emergent control strategy where nodes "lock down" their neighbors to isolate toxins based on pre-emptive warning signals.
  * **Bio-Realistic Constraints:**
      * **Metabolism:** Nodes must burn energy to exist.
      * **Fatigue:** Signaling systems tire out, preventing infinite feedback loops.
      * **Repeater Logic:** Root nodes act as active amplifiers for weak signals.

-----

## Installation

This project requires **Python 3.8+** and **PyTorch**.

-----

## Quick Start & Demos

The repository includes three distinct simulations.
**Note:** These scripts run in **Headless Mode** by default (printing telemetry to the console) to ensure compatibility with server environments and Python installations lacking GUI support (e.g., macOS/Homebrew).

### 1\. The "Paul Revere" Protocol (Single Plant)

Simulates a single 15-node plant. A "toxin" (Energy Spike) is injected into the top leaf. The plant detects the stress, screams, and locks down the stem nodes *before* the toxin reaches the roots.

```bash
python plant_sim_gated.py
```

  * **What to watch:** Look at the **`LOCKED`** column in the console.
  * **Success:** It will sit at `0` initially. Immediately after the `!!! BITE !!!` event (Tick 20), you will see the `LOCKED` count jump (`1 -> 2 -> 3...`) as the signal propagates down the stem.

### 2\. The Forest Wide Web (Ecosystem)

Simulates **two separate plants** connected only by a **Fungal Bridge** (Mycorrhizal Network) between their roots.

  * **Plant A** is bitten.
  * **Plant A** screams.
  * The signal crosses the bridge.
  * **Plant B** locks down in anticipation, despite having no physical contact with the threat.

<!-- end list -->

```bash
python forest_sim.py
```

  * **What to watch:** Look at the **`BRIDGE SIGNAL`** and **`PLANT B LOCKED`** columns.
  * **Success:** `BRIDGE SIGNAL` will rise after Tick 20. Shortly after, `PLANT B LOCKED` will increase (e.g., from `0` to `7` or `10`), proving inter-species communication.

### 3\. The Chaos Test (Robustness Validation)

Proves the theory is robust to noise and randomness, not just a scripted animation.

  * **Random Topology:** Generates a unique organic tree structure every run.
  * **Thermal Noise:** Injects random fluctuations ($\pm 1\%$) into energy and signal levels.
  * **Random Trauma:** Applies a bite at a random time with random force.

<!-- end list -->

```bash
python plant_sim_chaos.py
```

  * **Success Criteria:** Despite the random tree shape and noise, the `LOCKED` count must **always** spike immediately after the bite event. Run this multiple times to verify the behavior emerges consistently.

-----

## Project Structure

```text
.
├── det20_model.py             # THE ENGINE: Contains the DETSystem class and physics rules.
├── plant_sim_gated.py         # DEMO 1: Single plant active defense simulation.
├── forest_sim.py              # DEMO 2: Multi-plant ecosystem with fungal bridge.
├── plant_sim_chaos.py         # DEMO 3: Robustness testing with random topology & noise.
└── README.md                  # This file.
```

-----

## Mathematical Foundation

The system evolves state vector $\mathbf{S}_i = [F, \sigma, I, g]$ for every node $i$:

### 1\. The Active Repeater Equation (Information)

Nodes act as relay stations. If a signal is weak (whisper), specific nodes (Roots) amplify it.

$$I_i(t+1) = \text{Decay}(I_i) + \text{Diffusion}(\Delta I) + \text{Boost}(I_{in})$$

### 2\. The Metabolically Gated Flow (Energy)

Energy moves down gradients but is stopped by the Gating Factor $g$, which is controlled by the Information layer.

$$J_{i \to j} = \alpha_E \cdot \text{ReLU}(F_i - F_j) \cdot (\mathbf{A}_E)_{ij} \cdot g_i \cdot g_j$$

$$F_i(t+1) = \left[ F_i(t) - J_{out} + J_{in}(1-\gamma) \right] \cdot (1 - \mu)$$

Where $\gamma$ is entropy (friction) and $\mu$ is metabolic cost.

-----

## Novelty vs. Synthesis

DET 2.0 is a **Mesoscale Synthesis**. It bridges the gap between abstract Game Theory and detailed Molecular Dynamics.

  * **Reframing:** It utilizes established principles from **Reaction-Diffusion Systems** (Turing, 1952) and **Active Inference** (Friston, 2010).
  * **Novelty:** It operationalizes these into a lightweight "Physics Engine for Biology" ($<200$ lines of code) that allows for the real-time simulation of **Smart Fluids**—networks that physically reconfigure themselves based on information flow.

-----

## Roadmap

  * [x] **v1.0:** Basic Thermodynamic Consistency.
  * [x] **v2.0:** Active Gating & "Paul Revere" Protocol.
  * [x] **v3.0:** Ecosystems & Mycorrhizal Bridges.
  * [x] **v3.1:** Chaos & Robustness Testing (Headless Mode).
  * [ ] **v4.0:** Evolutionary Dynamics (Node Death & Reproduction).
  * [ ] **v5.0:** 3D Spatial Embedding.

## Contributing

Contributions are welcome\! Please open an issue to discuss proposed changes or submit a Pull Request.

**Citation:**
If you use DET 2.0 in your research, please cite:

> *Deep Existence Theory 2.0: A Dimensionally Consistent Framework for Distributed Bio-Intelligence (2025).*
