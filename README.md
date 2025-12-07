# OpenSystemThermodynamicNetwork (DET 2.0) / Unified Research Repository

## Project Mission  
This repository unifies two interrelated research threads — a physical / “open-universe” thread exploring a generalized thermodynamic network model of reality, and a computational / AI & comp-sci thread exploring algorithmic, simulation, and AI-driven extensions.  Our aim: not to claim ownership, but to contribute — to open up a flexible framework that others (and future us) can use, extend, and build upon, in service of expanding understanding of complex systems and “improving the universe/existence.”

---

## Core Mathematical Model (High-Level Overview)

At the heart of the “open-universe” thread is a representation of a system as a directed graph of **nodes** (active agents or subsystems) plus a special **reservoir node** representing an infinite external potential bath. Key ideas:

- Each node \(i\) holds a *free-level* \(F_i(t)\), representing its resource capacity (energy, information, “time-availability”, or a composite thereof).  
- Resource flows consist of **inter-node fluxes** \(J_{i \to j}(t)\) and **reservoir-to-node coupling** \(J_{\mathrm{res} \to i}(t)\).  
- Discrete-time update (tick \(k\) to \(k+1\)) uses integrated flows: outgoing flows reduce \(F_i\); incoming flows (from other nodes or reservoir) increase \(F_i\).  
- Reservoir coupling is potential-driven: nodes draw from the external bath when their \(F_i\) is below the reservoir potential \(\Phi_{\mathrm{res}}\), scaled by a conductivity factor \(\sigma_i\) and activity parameter \(a_i\).  
- Optionally, nodes adapt conductivity \(\sigma_i\) over time, based on their recent efficiency (ratio of incoming to outgoing flux), allowing dynamic reconfiguration of connectivity / flow capacity.  

In formula form (discrete tick):  

G_i^res = a_i · σ_i · max(0, Φ_res – F_i) · Δt
F_i(next) = F_i – γ·(outgoing flux) + (incoming from other nodes) + G_i^res
σ_i may adapt over time based on flux efficiency.

This model — combining energy/entropy-like flows, information/time resources, and adaptive coupling — aims to provide a broad, unified framework for phenomena from physical processes to computational/organizational dynamics.

---

## Repository Structure & Reading Paths  

- **Thermodynamics / Open-Universe** — conceptual, theoretical, simulation code, and experiments:  
  → See [`open_universe/README.md`](open_universe/README.md)  

- **AI / Computational-CS Thread** — AI models, computational experiments, data-driven analyses, algorithms:  
  → See [`ai_comp_sci/README.md`](ai_comp_sci/README.md)  

- **Shared & Utils** — generic support code, math utilities, helpers, cross-cutting tools and modules  

---

## Getting Started / Contribution & Use Guidelines  

1. Clone the repository.  
2. Choose the thread you want to explore (open_universe or ai_comp_sci).  
3. Follow the README in that sub-folder to set up dependencies, run simulations or experiments, and reproduce results.  
4. Feel free to experiment, extend, adapt — this project is open to all.  

Contributions, forks, and alternative implementations are welcome. If you build something novel, please document and share so the user-community can collectively grow this “network of possibility.”  

---

## License  

This project is released under **CC0 1.0 Universal** — public-domain dedication. Use, modify, and redistribute freely, for any purpose.  
See [`LICENSE`](LICENSE) for full text.  

---

## Acknowledgements & Philosophy  

This isn’t “my code.” It is an invitation — to thinkers, dreamers, engineers, scientists, explorers.  
If you use or build on this, consider it part of a shared journey to explore emergent complexity, open systems, and maybe a deeper understanding of “what could be.”  

