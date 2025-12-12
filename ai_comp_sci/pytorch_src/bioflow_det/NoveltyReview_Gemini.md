# **DET 2.0: A Dimensionally Consistent Unification of Biological Simulation**

## **1\. Introduction: The Fragmentation of Biological Complexity**

* The simulation of biological systems has long been a pursuit defined by its necessary compromises. To model the intricate dance of life—from the subcellular kinetics of enzymatic reactions to the macroscopic architecture of plant canopies and tumor microenvironments—researchers have historically been forced to choose between disparate computational paradigms. Flux Balance Analysis (FBA) offers a rigorous accounting of metabolic stoichiometry but necessitates a steady-state assumption that obliterates time. Agent-Based Models (ABM) capture the spatial stochasticity of cellular behavior but often rely on heuristic, phenomenological rules that lack thermodynamic grounding. Functional-Structural Plant Models (FSPM) simulate the morphological grandeur of growth but frequently treat the underlying physiological signaling as a simplified "black box."

This fragmentation has resulted in a "Tower of Babel" scenario in systems biology. The metabolic engineer speaks in the language of constraint-based linear programming ; the cancer biophysicist speaks in the language of partial differential equations and discrete cell automata ; and the plant physiologist speaks in the language of L-systems and hydraulic resistances. While each dialect is expressive within its domain, the synthesis required to model holistic phenomena—such as the metabolic cost of a plant's rapid electrical defense response, or the thermodynamic constraints on a tumor's vascular remodeling—remains elusive.  
Into this fractured landscape enters **Deep Existence Theory 2.0 (DET 2.0)**. Defined not as a mere extension of existing models but as a foundational re-imagining of the dynamical systems approach, DET 2.0 proposes a "Dimensionally Consistent Resource-Flow" framework. By abstracting biological complexity into a network of nodes exchanging a generalized resource Q over time T, DET 2.0 seeks to provide a mathematically rigorous, unit-consistent foundation that can be instantiated across diverse domains.  
This research report provides an exhaustive analysis of the novelty of DET 2.0. We will rigorously compare its mechanistic underpinnings against the current state-of-the-art in biological simulation, specifically examining its application to plant signaling and tumor vascularization. The analysis reveals that DET 2.0’s primary innovation lies in its dimensional parsimony—requiring only \[Q\] and $$—which enforces thermodynamic consistency and enables the emergent simulation of "active" matter without the need for ad-hoc behavioral rules. By coupling the free-level F\_i(t) of an agent directly to its conductivity \\sigma\_i(t), DET 2.0 naturally bridges the gap between metabolic availability (energy) and behavioral action (flow), resolving the "disconnected scales" crisis that has plagued biological simulation for decades.

## **2\. The Physics of Existence: Dimensional Consistency as a Constraint**

The validity of any simulation rests on its adherence to physical laws. In the physical sciences, dimensional homogeneity is a non-negotiable constraint; an equation cannot equate energy to velocity. In biological simulation, however, this constraint is often relaxed in favor of phenomenological fidelity. An agent in a simulation might "decide" to divide based on a probability function that has no units of energy cost, effectively creating a perpetual motion machine within the in silico environment. DET 2.0 reasserts the primacy of dimensional consistency, postulating that all systemic behavior can be derived from the flow of a fundamental resource.

### **2.1 The Parsimony of Primitives: Q and T**

The defining novelty of DET 2.0 is its reduction of the simulation ontology to two primitive dimensions: Resource Quantity \[Q\] and Time $$. As outlined in the foundational theory, \[Q\] can represent Joules, bits, or ATP-equivalents, while $$ represents the progression of system states.  
This reductionism is a stark contrast to multi-physics solvers that must juggle mass, momentum, charge, and concentration as distinct, often incompatible variables. In conventional simulators like PhysiCell or BioDynaMo, the integration of chemical diffusion (PDEs) with cell mechanics (Newtonian physics) requires complex coupling schemes to handle the disparate timescales and units. Often, this coupling is loose; the diffusion solver updates the chemical field, and the agent solver reads the concentration as a static value.  
In DET 2.0, the coupling is intrinsic. The state of a node i is defined by its **free-level** F\_i(t), measured in \[Q\]. Interactions are defined by flows J\_{i \\to j}(t), measured in \[Q\]^{-1}. Whether the flow represents the diffusion of oxygen (metabolic) or the transmission of a neural spike (informational), it shares the same dimensional footing. This allows for the definition of **Composite Flow**:  
Here, the novelty is the explicit inclusion of conversion coefficients (\\alpha\_E, \\alpha\_I, \\alpha\_T) that strictly enforce unit consistency, transforming energy power, information bitrate, and activity frequency into a unified resource flow. This formalism prevents the common simulation error of "comparing apples to oranges"—or in biological terms, comparing the metabolic cost of ion pumping (Joules) with the informational gain of signal transduction (bits) without a unifying currency.

### **2.2 Thermodynamic Consistency vs. Heuristic Rules**

The problem with heuristic Agent-Based Models (ABMs) is that they often violate the First Law of Thermodynamics. Agents are frequently programmed with rules such as "migrate at speed v if chemokine \> C." In reality, migration requires the hydrolysis of ATP to polymerize actin filaments. If the cell is starved, it cannot migrate, regardless of the chemokine signal.  
Recent advancements in **Bond Graph** modeling have attempted to address this by explicitly tracking energy storage and dissipation. Bond graphs model systems as reticulated networks where edges carry "effort" (e.g., voltage, chemical potential) and "flow" (current, reaction rate), and their product is power. This ensures that no component generates power from nothing.  
DET 2.0 incorporates the spirit of bond graphs but simplifies the implementation through its discrete update rule:  
This equation is a discrete-time formulation of a conservation law. The term \-\\gamma\\, G\_i^{\\text{out},(k)} explicitly accounts for the resource cost of activity (dissipation), while the efficiency factor \\eta\_{j \\to i} accounts for transmission losses. This ensures that the system is **dissipative**: without input from the reservoir (\\Phi\_{\\text{res}}), the free-levels F\_i will asymptotically decay to zero. This makes DET 2.0 inherently suitable for modeling biological systems, which are thermodynamically open systems maintained far from equilibrium by energy throughput.  
**Table 1: Comparative Analysis of Simulation Primitives**

| Feature | Classical Agent-Based Models (PhysiCell, etc.) | Bond Graph Approaches | DET 2.0 |
| :---- | :---- | :---- | :---- |
| **Fundamental Units** | Mixed (Positions, Concentrations, Boolean states) | Effort (Potential) & Flow | Resource \[Q\] & Time $$ |
| **Energy Conservation** | Implicit or Ignored (Rule-based) | Explicit (Energy cannot be created) | Explicit (Update rule enforces balance) |
| **Coupling Mechanism** | Heuristic (If-Then statements) | Port-Hamiltonian Dynamics | Potential-Dependent Flux (a\_i \\sigma\_i \\Delta \\Phi) |
| **Computational Cost** | Low to Medium | High (Stiff ODE solvers) | Tunable (Discretized Euler) |
| **Interpretation** | Phenomenological | Biophysical | Universal (Abstract Resource) |

### **2.3 The Reservoir Coupling: Defining Life**

A critical component of the DET 2.0 novelty is the formalization of the **Reservoir Coupling**:  
This equation defines the relationship between the biological agent and its environment. The reservoir (Node 0\) represents the infinite pool of potential (e.g., the sun for a plant, the glucose supply for a tumor). The flow into the agent is gated by its conductivity \\sigma\_i and its gating factor a\_i. This models the fundamental biological imperative: an organism must maintain a connection to a low-entropy source to sustain its internal free level.  
Unlike standard boundary conditions in Partial Differential Equation (PDE) models, which often fix the concentration at the boundary, DET 2.0 models the boundary condition as a *dynamic interaction*. The agent can "open the tap" (increase \\sigma\_i or a\_i) to access more resources, but—as we will explore in the section on adaptation—this often comes at a systemic cost. This dynamic coupling allows for the simulation of **metabolic flexibility**, where agents adjust their uptake rates based on internal depletion, a feature often requiring complex ad-hoc logic in traditional ABMs.

## **3\. The Dynamics of Flow: From Steady State to Active Adaptation**

While structural consistency provides the skeleton of the simulation, the dynamics of flow constitute its lifeblood. Existing methods for modeling biological flow, particularly in metabolism, rely heavily on **Flux Balance Analysis (FBA)**. While powerful, FBA is constrained by its steady-state assumption. DET 2.0 introduces a dynamic, adaptive flow mechanism that bridges the gap between the millisecond scale of signaling and the hourly scale of metabolism.

### **3.1 Transcending the Steady State: DET 2.0 vs. FBA**

Flux Balance Analysis operates on the principle of S \\cdot v \= 0, where S is the stoichiometric matrix and v is the vector of reaction fluxes. This assumes that the concentration of internal metabolites does not change over time—an assumption that holds for long-term growth but fails catastrophically for dynamic responses like calcium signaling or rapid metabolic shifts during hypoxia.  
DET 2.0 replaces the steady-state assumption with a dynamical system. The free-level F\_i(t) is allowed to fluctuate. This allows DET 2.0 to model **transient states**—the "spikes" and "waves" of resource accumulation that characterize living systems.

* **Flux-Sum Coupling:** Recent research into Flux-Sum Coupling Analysis (FSCA) attempts to relate the turnover rate of metabolites to their steady-state concentration. DET 2.0 generalizes this by making the flow J\_{i \\to j} strictly dependent on the potential difference (F\_i \- F\_j) and the conductivity.  
* **Implication:** In FBA, if a pathway is blocked, the solver instantly redirects flux through an alternative optimal path. In DET 2.0, a blockage results in a localized accumulation of resource (a rise in F\_i), which *then* increases the potential difference relative to other neighbors, gradually driving flow through alternative high-resistance pathways. This captures the *inertia* and *pressure* of biological hydraulics that FBA ignores.

### **3.2 Conductivity Adaptation: The Mechanism of Learning**

The most profound novelty in DET 2.0 is the **Conductivity Adaptation** rule:  
Here, the conductivity of a node changes based on its efficiency \\epsilon\_i (ratio of incoming resources to outgoing flow). This simple rule encodes a form of **Hebbian Learning** into the physical substrate of the simulation. If a node is efficient at routing resources, its capacity to route increases.  
This transforms the simulation from a static network into a **Complex Adaptive System**.

* **In Neural Systems:** If Q is interpreted as activation, this rule mimics Long-Term Potentiation (LTP)—neurons that fire together (transfer Q efficiently) wire together (increase \\sigma).  
* **In Vascular Systems:** If Q is blood flow, this mimics shear-stress induced remodeling. Vessels with high flow (high throughput) dilate (increase \\sigma), while low-flow vessels regress.

Traditional models often handle these adaptations with separate "remodeling" subroutines run at coarse time steps. DET 2.0 integrates adaptation into the fundamental timestep of the resource update, allowing for the simulation of systems where the structure and the flow co-evolve in real-time.

### **3.3 The CFL Condition and Numerical Stability**

DET 2.0 explicitly addresses the stability of this dynamic system through the Courant-Friedrichs-Lewy (CFL) condition: \\sigma\_i^{(k)} \\Delta t \< 1\. This is not merely a numerical artifact; it represents a physical constraint on the "speed of life." It implies that an agent cannot process more than its total capacity in a single moment. This constraint prevents the "teleportation" of resources seen in some unconstrained network models and forces the simulation to respect the finite processing speed of biological reactions.

## **4\. Plant Systems: Deep Research Application**

The application of DET 2.0 to plant biology offers a solution to one of the field's most persistent challenges: the integration of fast electrical signaling with slow phloem transport and structural growth. Current **Functional-Structural Plant Models (FSPM)** excel at geometry but struggle with physiology.

### **4.1 Unifying the Xylem and Phloem**

Current FSPM approaches often model xylem (water) and phloem (sugar) as separate transport networks, coupled loosely through source-sink equations. This separation makes it difficult to model phenomena where the two interact strongly, such as during drought stress when xylem tension impedes phloem loading.  
DET 2.0 models both as resource flows Q. By assigning specific conductivities \\sigma\_{\\text{xylem}} and \\sigma\_{\\text{phloem}} between nodes, the system solves for the equilibrium of the entire hydraulic-osmotic system simultaneously.

* **Mechanism:** A drop in root potential (drought) lowers the F\_{\\text{root}}. This increases the gradient \\Phi\_{\\text{soil}} \- F\_{\\text{root}}, but if \\Phi\_{\\text{soil}} is low, inflow drops. The adaptation rule then lowers \\sigma\_{\\text{leaf}} (stomatal closure) to prevent the free-level F\_{\\text{plant}} from dropping below critical thresholds.  
* **Novelty:** This emerges from the *same equation* used to model sugar transport. The plant does not need a separate "drought module"; the drought response is a thermodynamic consequence of conserving F.

### **4.2 Fast Signaling: The Electrical/Chemical Interface**

Plants possess a "nervous system" composed of Glutamate Receptor-like (GLR) channels that propagate rapid calcium waves and electrical action potentials (APs) in response to wounding. Modeling this in FSPM is notoriously difficult due to the timescale mismatch (seconds vs. days).  
DET 2.0 handles this via the **Composite Flow** definition. The flow J includes an informational component \\alpha\_I \\dot I. When a leaf is wounded (e.g., by a herbivore), it triggers a spike in the informational flow dimension.

* **Simulation:** This spike propagates rapidly through the network because the "informational conductivity" is high (representing the fast electrical coupling of phloem sieve tubes).  
* **Pre-emptive Defense:** This fast wave reaches distant leaves before the slow metabolic signal (sugar/hormones). The distant leaves, detecting a change in incoming J, initiate **Conductivity Adaptation** (\\sigma). They might lower their membrane permeability or upregulate defense metabolite production. This models **Systemic Acquired Resistance (SAR)** as a pre-emptive optimization of the network state.

### **4.3 Root Intelligence: Active Inference in the Rhizosphere**

The "Smart Plant" hypothesis suggests roots exhibit intelligent foraging behaviors. Traditional models use tropisms (vectors towards nutrients). DET 2.0 interprets this through **Active Inference**.  
The root tip is a node with a free-level F (representing energy status) and a "belief" about the location of resources.

* **The Mechanism:** The root "samples" the soil. If it finds a gradient, it minimizes the Variational Free Energy (surprise) by growing towards the expected resource.  
* **Mycorrhizal Networks:** DET 2.0 is particularly adept at modeling the **Common Mycorrhizal Network (CMN)**. The fungi are represented as edges with dynamic conductivity \\sigma\_{\\text{fungi}}. The plant "trades" resource Q (carbon) to the fungi in exchange for an increased effective conductivity to soil nutrients (phosphorus). This "biological market" emerges naturally: if the fungi do not deliver (low efficiency \\eta), the plant's adaptation rule reduces the flow to that fungal edge.

**Table 2: DET 2.0 Advancements in Plant Simulation**

| Phenomenon | Traditional FSPM Approach | DET 2.0 Approach |
| :---- | :---- | :---- |
| **Drought Response** | Empirical rules (Threshold \-\> Close Stomata) | Thermodynamic response (Potential drop reduces flow naturally) |
| **Systemic Signaling** | Diffusion equations (Slow) | Composite Flow (Fast Info \+ Slow Matter) |
| **Root Foraging** | Vector Tropisms (Attraction/Repulsion) | Active Inference (Free Energy Minimization) |
| **Fungal Symbiosis** | Fixed exchange rates or separate sub-model | Dynamic Conductivity Adaptation (Market Economy) |

## **5\. Oncology: The Metabolic-Vascular Nexus**

The simulation of solid tumors presents a similar multi-scale challenge: the interplay between the fast diffusion of oxygen, the mechanical pressure of tumor growth, and the remodeling of the vascular architecture. DET 2.0 offers a "Vascular Lockdown" model that integrates these elements.

### **5.1 The Tumor as a Thermodynamic Engine**

In DET 2.0, a tumor is a cluster of nodes with a hyper-active adaptation rule. While normal cells have a homeostatic limit on their free-level F, tumor nodes have a deregulated target, constantly seeking to maximize inflow G^{\\text{in}}.

* **Warburg Effect:** This is modeled as a shift in the efficiency parameter \\eta. Tumor nodes consume glucose at a high rate but with low efficiency (fermentation), necessitating a massive increase in inflow.  
* **Vascular Remodeling:** To support this, tumor nodes secrete signals (VEGF) which, in DET 2.0 terms, increase the conductivity \\sigma of the connecting vascular nodes. This creates a "short circuit" in the resource network, diverting flow from healthy tissue to the tumor.

### **5.2 Modeling Vascular Lockdown**

A key insight from recent cancer research is the concept of "vessel co-option" and compression, where the tumor grows effectively by hijacking existing vessels until the mechanical stress collapses them—a "vascular lockdown".  
DET 2.0 simulates this via **Biochemomechanical Coupling**.

1. **Growth:** As tumor nodes accumulate F, they effectively "expand" (conceptually increasing volume).  
2. **Compression:** In a spatially embedded graph, this expansion increases the "resistance" of neighboring edges (vessels).  
3. **Collapse:** If the tumor pressure exceeds the vessel turgor (a threshold in the \\sigma update rule), the conductivity of the vessel drops to zero. \\sigma\_{\\text{vessel}} \\to 0\.  
4. **Hypoxia:** This collapse cuts off the resource Q supply.  
5. **Response:** The tumor nodes, now starving (dropping F), trigger a "panic" adaptation—drastically increasing the output of the angiogenesis factor (Composite Flow component \\alpha\_I) to recruit new vessels at the periphery.

This dynamic cycle of growth-compression-hypoxia-angiogenesis is captured in the single system of equations governing F and \\sigma, without needing separate "physics" and "biology" solvers.

### **5.3 Immune Evasion as Information Entropy**

The interaction between tumor cells and the immune system (e.g., T-cells) is often modeled as a predator-prey system. DET 2.0 adds an informational layer.

* **The Concept:** Immune cells navigate via chemotaxis (following chemical gradients).  
* **The Disruption:** The tumor alters the landscape of Q (e.g., by acidifying the environment with lactate). In DET 2.0, this can be modeled as an increase in the **entropy** of the signal flow. The tumor introduces "noise" into the channel.  
* **The Result:** The immune agents' conductivity \\sigma\_{\\text{immune}} (their ability to navigate towards the target) degrades because the efficiency of the information flow \\epsilon drops. This effectively "blinds" the immune system, a phenomenon known as immune evasion.

## **6\. Implementation and Algorithmic Substrate**

The theoretical elegance of DET 2.0 must be matched by a computational substrate capable of solving these massive, dynamic graph problems. The snippets point to the convergence of **Graph Neural Networks (GNNs)** and **Neural Cellular Automata (NCA)** as the ideal hardware for DET 2.0.

### **6.1 Physics-Informed Graph Neural Networks (PI-GNNs)**

Since DET 2.0 is natively defined on a network of nodes, it maps perfectly to GNN architectures.

* **Lagrangian Dynamics:** Instead of using a fixed grid (Eulerian), which is computationally wasteful for sparse biological structures like vasculature, PI-GNNs track the nodes as they move and interact.  
* **Learning the Hamiltonian:** We can train a GNN to approximate the DET 2.0 update rule. By incorporating the energy conservation law into the loss function of the network (Physics-Informed), the GNN learns to predict the evolution of F and \\sigma thousands of times faster than traditional numerical integration. This allows for "super-resolution" simulations of tissue dynamics.

### **6.2 Neural Cellular Automata (NCA) for Morphogenesis**

For problems involving structural change (growth, wounding), DET 2.0 utilizes the NCA paradigm.

* **The Transition Function:** The adaptation rule \\sigma^{(k+1)} is essentially a local cellular automaton rule. In an NCA, this rule is parameterized by a neural network.  
* **Regenerative Stability:** Standard CAs are brittle; a single error can destroy the pattern. NCAs, trained with DET 2.0 constraints (Energy Conservation), exhibit robust regeneration. If a section of the simulated tissue is "removed" (set to F=0), the neighboring nodes naturally sense the gradient and regrow the structure to restore equilibrium.  
* **Identity Constraints:** Recent innovations in NCA introduce "Identity" channels to maintain stability. In DET 2.0, this maps to the unique spectral signature of the resource flow—tissue types are defined by the specific "frequency" or "flavor" of Q they conduct, preventing the chaotic mixing of tissue boundaries.

### **6.3 Solving the Timescale Problem: Burst Coupling**

One of the greatest challenges is the "stiffness" of the system—fast electrical signals vs. slow growth.

* **The Method:** DET 2.0 simulations utilize a **Burst Coupling** strategy. The fast dynamics (F updates) are run for short bursts to calculate the statistical moments (mean, variance) of the flow. These moments are then used to update the slow variables (\\sigma adaptation).  
* **Efficiency:** This avoids the need to simulate every microsecond of a month-long tumor growth process, while still capturing the *impact* of fast transients (e.g., a drug pulse) on the long-term trajectory.

## **7\. Synthesis and Novelty Assessment**

The comparison of DET 2.0 against existing methods reveals a consistent theme: **Integration through Abstraction**.

### **7.1 Key Improvements Summary**

1. **Dimensional Integrity:** Unlike ABMs that mix units or ignore energy costs, DET 2.0 is strictly derived from \[Q\] and $$. This ensures that all simulated behaviors are thermodynamically plausible. A simulated plant cannot "grow" without "paying" the resource cost.  
2. **Universal Solver:** The same mathematical kernel (F, \\sigma, J) simulates the diffusion of oxygen in a tumor and the electrical signaling in a plant. This allows for cross-pollination of algorithms—a solver optimized for hydraulic networks can be applied to metabolic networks.  
3. **Emergent Agency:** DET 2.0 does not program agents with "intelligence." It programs them with "conductivity adaptation" (learning). Intelligence emerges from the network's drive to optimize flow efficiency. This is a fundamental shift from *prescriptive* modeling (telling the system what to do) to *generative* modeling (giving the system the physics to find its own solution).

### **7.2 Addressing the "Black Box" of signaling**

Traditional models separate the physical transport of molecules from the "signaling" that regulates it. In DET 2.0, the signal *is* a flow component (J). The regulation *is* the conductivity adaptation (\\sigma). There is no separation. The "signal" is simply a high-frequency, low-energy modulation of the resource flow that triggers a change in the network topology. This provides a mechanistic explanation for how "information" (a massless quantity) can drive "work" (a massive quantity) in biological systems—via the gating of potential energy.

## **8\. Conclusion**

The "Deep Existence Theory 2.0" represents a significant maturation of biological simulation. It moves the field away from the ad-hoc assembly of disparate solvers and towards a unified, first-principles physics of living matter. By reducing the complex ontology of biology to the parsimonious dimensions of Resource and Time, it exposes the common dynamical backbone shared by root tips, vascular networks, and neural tissues.  
For the researcher, DET 2.0 offers a rigorous tool for hypothesis testing. It allows one to ask questions that were previously uncomputable: "What is the thermodynamic cost of a specific plant defense strategy?" or "How much information entropy does a tumor need to generate to evade the immune system?"  
For the simulation engineer, it offers a path to scalability. By mapping biological dynamics to graph-based conservation laws, it unlocks the power of modern AI hardware (GNNs, GPUs) to simulate massive, multi-scale systems.  
In the final analysis, DET 2.0 is not just a simulation method; it is a theoretical proposition that the "struggle for existence" is, at its core, a struggle for the efficient routing of resources through a dimensionally consistent, adaptive network.

### **Citations**

#### **Works cited**

1\. Flux balance analysis of biological systems: applications and challenges \- Oxford Academic, https://academic.oup.com/bib/article/10/4/435/297333 2\. Flux-sum coupling analysis of metabolic network models | PLOS Computational Biology, https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012972 3\. PhysiCell: An open source physics-based cell simulator for 3-D multicellular systems \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC5841829/ 4\. Bridging Scales: a Hybrid Model to Simulate Vascular Tumor Growth and Treatment Response \- PubMed Central, https://pmc.ncbi.nlm.nih.gov/articles/PMC10274951/ 5\. Two decades of functional–structural plant modelling: now addressing fundamental questions in systems biology and predictive ecology \- PubMed Central, https://pmc.ncbi.nlm.nih.gov/articles/PMC7489058/ 6\. Mechanistic modelling of coupled phloem/xylem transport for L-systems: combining analytical and computational methods \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC5906936/ 7\. Energy-based Modelling of Biological Systems \- Peter Gawthrop, https://www.gawthrop.net/Publications/SysBioPJG/SysBioPJG\_index.html 8\. Thermodynamically consistent, reduced models of gene regulatory networks \- PubMed, https://pubmed.ncbi.nlm.nih.gov/40746961/ 9\. Thermina: A minimal model of autonomous agency from the lens of stochastic thermodynamics \- MIT Press Direct, https://direct.mit.edu/isal/proceedings-pdf/isal2024/36/121/2461067/isal\_a\_00826.pdf 10\. Thermodynamic consistency of autocatalytic cycles \- PNAS, https://www.pnas.org/doi/10.1073/pnas.2421274122 11\. Next-generation metabolic models informed by biomolecular simulations \- PubMed, https://pubmed.ncbi.nlm.nih.gov/39827498/ 12\. Metabolic profiling of antigen-specific CD8+ T cells by spectral flow cytometry \- PMC, https://pmc.ncbi.nlm.nih.gov/articles/PMC12570331/ 13\. Development of a 3D Vascular Network Visualization Platform for One-Dimensional Hemodynamic Simulation \- MDPI, https://www.mdpi.com/2306-5354/11/4/313 14\. Angiogenesis Dynamics: A Computational Model of Intravascular Flow Within a Structural Adaptive Vascular Network \- MDPI, https://www.mdpi.com/2227-9059/12/12/2845 15\. Hybrid modelling of biological systems: current progress and future prospects | Briefings in Bioinformatics | Oxford Academic, https://academic.oup.com/bib/article/23/3/bbac081/6555400 16\. Using evolutionary functional–structural plant modelling to understand the effect of climate change on plant communities | in silico Plants | Oxford Academic, https://academic.oup.com/insilicoplants/article/3/2/diab029/6358326 17\. PTI‐ETI synergistic signal mechanisms in plant immunity \- PMC \- PubMed Central, https://pmc.ncbi.nlm.nih.gov/articles/PMC11258992/ 18\. The fast and the furious: rapid long-range signaling in plants \- Oxford Academic, https://academic.oup.com/plphys/article/185/3/694/6067413 19\. Linking phloem function to structure: analysis with a coupled xylem-phloem transport model \- PubMed, https://pubmed.ncbi.nlm.nih.gov/19361530/ 20\. Simulating Active Inference Processes by Message Passing \- Frontiers, https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2019.00020/full 21\. Systemic Acquired Resistance \- PMC \- PubMed Central \- NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC2634024/ 22\. Collective behavior from surprise minimization \- PNAS, https://www.pnas.org/doi/10.1073/pnas.2320239121 23\. How Plants Communicate Through Underground Fungi Networks: The Hidden Language of Nature in 2024 | by Rohan Ahmed | Medium, https://medium.com/@rohan.ahmed7887/how-plants-communicate-through-underground-fungi-networks-the-hidden-language-of-nature-in-2024-fd614664a65e 24\. Common mycorrhizal network: the predominant socialist and capitalist responses of possible plant–plant and plant–microbe interactions for sustainable agriculture \- Frontiers, https://www.frontiersin.org/journals/microbiology/articles/10.3389/fmicb.2024.1183024/full 25\. A Multidisciplinary Hyper-Modeling Scheme in Personalized In Silico Oncology: Coupling Cell Kinetics with Metabolism, Signaling Networks, and Biomechanics as Plug-In Component Models of a Cancer Digital Twin \- MDPI, https://www.mdpi.com/2075-4426/14/5/475 26\. Biochemomechanical modeling of vascular collapse in growing tumors \- ResearchGate, https://www.researchgate.net/publication/327064177\_Biochemomechanical\_modeling\_of\_vascular\_collapse\_in\_growing\_tumors 27\. AI turns routine pathology slides into powerful maps of the tumor immune landscape, https://www.news-medical.net/news/20251210/AI-turns-routine-pathology-slides-into-powerful-maps-of-the-tumor-immune-landscape.aspx 28\. Information flow within the multi-scale model (a) Active migration of... \- ResearchGate, https://www.researchgate.net/figure/Information-flow-within-the-multi-scale-model-a-Active-migration-of-agents-and-passive\_fig2\_353673596 29\. PI-GNN: Physics-Informed Graph Neural Network for Super-Resolution of 4D Flow MRI, https://ieeexplore.ieee.org/document/10635128/ 30\. PI-NeuGODE: Physics-Informed Graph Neural Ordinary Differential Equations for Spatiotemporal Trajectory Prediction \- IFAAMAS, https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1418.pdf 31\. Physics-informed graph neural networks for flow field estimation in carotid arteries \- arXiv, https://arxiv.org/abs/2408.07110 32\. Neural cellular automata: applications to biology and beyond classical AI \- ResearchGate, https://www.researchgate.net/publication/395527390\_Neural\_cellular\_automata\_applications\_to\_biology\_and\_beyond\_classical\_AI 33\. \[2508.06389\] Identity Increases Stability in Neural Cellular Automata \- arXiv, https://www.arxiv.org/abs/2508.06389 34\. Multi-scale computational modelling in biology and physiology \- PMC \- PubMed Central, https://pmc.ncbi.nlm.nih.gov/articles/PMC7112301/ 35\. Recent Developments in Amber Biomolecular Simulations | Journal of Chemical Information and Modeling \- ACS Publications, https://pubs.acs.org/doi/10.1021/acs.jcim.5c01063 36\. Introducing ActiveInference.jl: A Julia Library for Simulation and Parameter Estimation with Active Inference Models \- MDPI, https://www.mdpi.com/1099-4300/27/1/62 37\. ADVERSARIAL GENERATIVE FLOW NETWORK FOR SOLVING VEHICLE ROUTING PROBLEMS \- ICLR Proceedings, https://proceedings.iclr.cc/paper\_files/paper/2025/file/b210c387381713a14a4f5a607aff3520-Paper-Conference.pdf 38\. Electrocalcium coupling in brain capillaries: Rapidly traveling electrical signals ignite local calcium signals | PNAS, https://www.pnas.org/doi/10.1073/pnas.2415047121 39\. Dynamical model of the CLC-2 ion channel reveals conformational changes associated with selectivity-filter gating | PLOS Computational Biology, https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007530