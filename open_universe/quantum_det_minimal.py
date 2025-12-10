"""
Quantum DET 2.1 minimal prototype (v2)

Adds:
- Tracking of reservoir *entanglement* via reduced single-qubit entropies
- Keeps the same 2-node, 2-qubit Bell-state setup
- DET-style free-level updates with simple consistency feedback

Run:
    python quantum_det_minimal_v2.py
"""

import numpy as np


# ---------- Utility functions ----------

def kron_n(*ops):
    """Kronecker product of multiple operators."""
    result = np.array([[1.0]], dtype=complex)
    for op in ops:
        result = np.kron(result, op)
    return result


def projector_z(num_qubits, qubit_index, outcome):
    """
    Projector Π_i(o) for measuring qubit 'qubit_index' in Z basis with outcome 0 or 1.
    qubit_index: 0-based (0 = most significant qubit in our convention).
    """
    # Single-qubit projectors
    P0 = np.array([[1, 0],
                   [0, 0]], dtype=complex)
    P1 = np.array([[0, 0],
                   [0, 1]], dtype=complex)
    P = P0 if outcome == 0 else P1

    ops = []
    for q in range(num_qubits):
        if q == qubit_index:
            ops.append(P)
        else:
            ops.append(np.eye(2, dtype=complex))
    return kron_n(*ops)


def von_neumann_entropy(rho, tol=1e-12):
    """Compute von Neumann entropy S = -Tr(rho log2 rho)."""
    # Ensure Hermitian
    rho = 0.5 * (rho + rho.conj().T)
    vals, _ = np.linalg.eigh(rho)
    vals = np.real(vals)
    vals = vals[vals > tol]
    if len(vals) == 0:
        return 0.0
    return float(-np.sum(vals * np.log2(vals)))


# ---------- Classes ----------

class Node:
    def __init__(self, node_id, F0=0.0, sigma0=1.0, a0=1.0, beta0=0.1):
        self.node_id = node_id
        self.k = 0  # local event index
        self.F = float(F0)
        self.sigma = float(sigma0)
        self.a = float(a0)
        self.beta = float(beta0)
        self.record = []  # list of (outcome, global_index)

    def rate(self):
        return max(self.sigma * self.a, 0.0)

    def add_event(self, outcome, K):
        self.record.append((outcome, K))
        self.k += 1


class QuantumReservoir:
    def __init__(self, num_qubits=2):
        self.num_qubits = num_qubits
        self.rho = self._initial_state()

    def _initial_state(self):
        """
        Start in a Bell-like entangled state for 2 qubits:
        |ψ> = (|00> + |11>)/sqrt(2)
        """
        if self.num_qubits != 2:
            # For simplicity, start in maximally mixed state for other sizes
            dim = 2 ** self.num_qubits
            return np.eye(dim, dtype=complex) / dim

        zero = np.array([[1], [0]], dtype=complex)
        one = np.array([[0], [1]], dtype=complex)
        psi = (np.kron(zero, zero) + np.kron(one, one)) / np.sqrt(2.0)
        rho = psi @ psi.conj().T
        return rho

    def measure_and_collapse(self, qubit_index, theta=0.0):
        """
        Measure qubit_index in a rotated basis, return (outcome, probability_before).
        Collapse rho accordingly.
        """
        # Build local single-qubit projectors in a rotated basis.
        # theta = 0.0 -> standard Z measurement.
        Pz0 = np.array([[1, 0],
                        [0, 0]], dtype=complex)
        Pz1 = np.array([[0, 0],
                        [0, 1]], dtype=complex)

        if abs(theta) < 1e-12:
            # No rotation: standard Z basis
            P0_local, P1_local = Pz0, Pz1
        else:
            # Rotate measurement axis around Y by angle theta
            c = np.cos(theta / 2.0)
            s = np.sin(theta / 2.0)
            U = np.array([[c, -s],
                          [s,  c]], dtype=complex)
            P0_local = U @ Pz0 @ U.conj().T
            P1_local = U @ Pz1 @ U.conj().T

        # Embed into full Hilbert space for the selected qubit
        probs = []
        projectors = []
        for outcome in [0, 1]:
            local_P = P0_local if outcome == 0 else P1_local
            ops = []
            for q in range(self.num_qubits):
                if q == qubit_index:
                    ops.append(local_P)
                else:
                    ops.append(np.eye(2, dtype=complex))
            P = kron_n(*ops)
            projectors.append(P)
            p = np.real(np.trace(P @ self.rho))
            probs.append(max(p, 0.0))

        # Normalize probabilities (robust to small numerical errors)
        total_p = sum(probs)
        if total_p <= 0:
            # Fallback: treat as uniform
            probs = [0.5, 0.5]
        else:
            probs = [p / total_p for p in probs]

        outcome = np.random.choice([0, 1], p=probs)
        P = projectors[outcome]
        p_out = probs[outcome]

        # Collapse
        num = P @ self.rho @ P
        if p_out > 0:
            self.rho = num / p_out
        else:
            # Numerical guard: if p_out ~ 0, fallback to renormalized num
            trace_num = np.real(np.trace(num))
            self.rho = num / max(trace_num, 1e-12)

        return outcome, p_out

    def reduced_density(self, qubit_index):
        """
        Reduced density matrix for a single qubit in a 2-qubit system (num_qubits=2).
        qubit_index: 0 or 1.
        """
        assert self.num_qubits == 2, "reduced_density is implemented for 2 qubits only."
        rho = self.rho
        dim = 4  # 2 qubits

        # Basis ordering: |q0 q1> with q0 as most significant bit.
        # Trace out the *other* qubit.
        if qubit_index == 0:
            # Trace out qubit 1 (LSB)
            # ρ0[i,j] = Σ_k ρ[2*i + k, 2*j + k]
            rho_red = np.zeros((2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        rho_red[i, j] += rho[2*i + k, 2*j + k]
        else:
            # Trace out qubit 0 (MSB)
            # ρ1[i,j] = Σ_k ρ[2*k + i, 2*k + j]
            rho_red = np.zeros((2, 2), dtype=complex)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        rho_red[i, j] += rho[2*k + i, 2*k + j]

        return rho_red

    def apply_dephasing(self, p):
        """
        Apply a simple global depolarizing-like noise to the reservoir:
            ρ -> (1 - p) ρ + p * I / dim
        where dim = 2**num_qubits.
        This guarantees an increase in mixedness for any non-maximally mixed state.
        """
        if p <= 0.0:
            return

        dim = 2 ** self.num_qubits
        I = np.eye(dim, dtype=complex) / dim
        self.rho = (1.0 - p) * self.rho + p * I


class QuantumDetNetwork:
    def __init__(self, num_nodes=2, Phi_res=0.0, lambda_step=0.1,
                 noise_std=0.01, dephasing_p=0.01, alpha_rate=0.0, seed=None):
        assert num_nodes == 2, "This minimal prototype assumes 2 nodes = 2 qubits."
        if seed is not None:
            np.random.seed(seed)

        self.num_nodes = num_nodes
        self.nodes = [Node(i) for i in range(num_nodes)]
        self.reservoir = QuantumReservoir(num_qubits=num_nodes)
        self.Phi_res = float(Phi_res)
        self.lambda_step = float(lambda_step)
        self.noise_std = float(noise_std)
        self.dephasing_p = float(dephasing_p)
        self.alpha_rate = float(alpha_rate)
        self.K = 0  # global event index

        # History for analysis
        self.history_F = []            # list of [F_0, F_1, ...]
        self.history_S_full = []       # von Neumann entropy of full reservoir state
        self.history_S_local = []      # [S(ρ_0), S(ρ_1)] entropies of single-qubit reductions
        self.history_events = []       # (K, node_id, outcome)

    def _log_state(self):
        """Log current F values and reservoir entropies."""
        # Log F
        self.history_F.append([n.F for n in self.nodes])

        # Full reservoir entropy (will be 0 if state is pure)
        S_full = von_neumann_entropy(self.reservoir.rho)
        # Local single-qubit entropies (track entanglement / mixedness)
        rho0 = self.reservoir.reduced_density(qubit_index=0)
        rho1 = self.reservoir.reduced_density(qubit_index=1)
        S0 = von_neumann_entropy(rho0)
        S1 = von_neumann_entropy(rho1)

        self.history_S_full.append(S_full)
        self.history_S_local.append([S0, S1])

        # Log initial state (K = 0) before any events
        # This call moved here to ensure initial entropies are recorded
    def _choose_node(self):
        """
        Choose next node to update.
        Event rates are DET-modulated:
            r_i = σ_i * a_i * (1 + alpha_rate * |F_i - Phi_res|)
        so nodes farther from the reservoir potential couple more often.
        """
        rates = []
        for n in self.nodes:
            base = n.rate()
            det_factor = 1.0 + self.alpha_rate * abs(n.F - self.Phi_res)
            rates.append(max(base * det_factor, 0.0))
        rates = np.array(rates, dtype=float)
        total = rates.sum()
        if total <= 0:
            # fallback: uniform choice
            probs = np.ones(self.num_nodes) / self.num_nodes
        else:
            probs = rates / total
        idx = np.random.choice(range(self.num_nodes), p=probs)
        return self.nodes[idx]

    def _consistency_feedback(self):
        """
        Simple global consistency: push F_i toward the global average.
        C_i = F_i - mean(F).
        """
        F_vals = np.array([n.F for n in self.nodes], dtype=float)
        mean_F = float(F_vals.mean())
        return [n.F - mean_F for n in self.nodes]

    def step(self):
        """
        Perform one global event:
        - choose node by rate
        - measure corresponding qubit
        - update node free-level (DET-style)
        - collapse reservoir
        - log F and entropies
        """
        node = self._choose_node()
        i = node.node_id

        # Apply decoherence (global depolarizing noise) to the reservoir
        self.reservoir.apply_dephasing(self.dephasing_p)

        # Reservoir measurement on qubit i in an F-dependent rotated basis.
        # Here we use theta = F_i (in radians); F is typically small, so this is a gentle tilt.
        theta = 5.0 * node.F
        outcome, p_out = self.reservoir.measure_and_collapse(qubit_index=i, theta=theta)

        # Consistency feedback (global)
        C_vals = self._consistency_feedback()
        C_i = C_vals[i]

        # Classical reservoir pull term
        pull = node.a * node.sigma * (self.Phi_res - node.F) * self.lambda_step

        # Consistency correction (gradient-like)
        # We use a minus sign so that positive C_i pulls F_i back toward the global mean.
        consistency_term = -node.beta * C_i * self.lambda_step

        # Noise
        noise = np.random.normal(scale=self.noise_std)

        # Update free-level
        node.F = node.F + pull + consistency_term + noise

        # Record event
        node.add_event(outcome, self.K)

        # Log history after this event
        self._log_state()

        self.history_events.append((self.K, i, outcome))

        # Increment global index
        self.K += 1

    def run(self, num_events=100):
        for _ in range(num_events):
            self.step()

    def summary(self):
        print("=== Quantum DET minimal prototype (v2) summary ===")
        print(f"Global events (K): {self.K}")
        for n in self.nodes:
            print(f"Node {n.node_id}:")
            print(f"  F = {n.F:.4f}")
            print(f"  sigma = {n.sigma:.4f}")
            print(f"  a = {n.a:.4f}")
            print(f"  beta = {n.beta:.4f}")
            print(f"  events recorded = {len(n.record)}")

        S_full_arr = np.array(self.history_S_full)
        S_loc_arr = np.array(self.history_S_local)  # shape (K, 2)
        print("\nReservoir entropy (full state, bits):")
        print(f"  min={S_full_arr.min():.4f}, max={S_full_arr.max():.4f}, final={S_full_arr[-1]:.4f}")
        print("Local single-qubit entropies (S(ρ_0), S(ρ_1), bits):")
        print(f"  initial: {self.history_S_local[0]}")
        print(f"  final:   {self.history_S_local[-1]}")


def main():
    net = QuantumDetNetwork(
        num_nodes=2,
        Phi_res=0.0,
        lambda_step=0.05,
        noise_std=0.02,
        dephasing_p=0.01,   # small decoherence per event
        alpha_rate=10.0,     # DET coupling: nodes far from Phi_res tick more often
        seed=42,
    )
    net._log_state()
    net.run(num_events=5000)
    
    F_arr = np.array(net.history_F)  # shape (K+1, 2)
    mean_abs_F0 = np.mean(np.abs(F_arr[:, 0]))
    mean_abs_F1 = np.mean(np.abs(F_arr[:, 1]))
    print(f"\nMean |F|: node 0 = {mean_abs_F0:.4f}, node 1 = {mean_abs_F1:.4f}")

    # Approximate mean rate factor
    alpha = net.alpha_rate
    Phi = net.Phi_res
    rate_factor0 = np.mean(1 + alpha * np.abs(F_arr[:, 0] - Phi))
    rate_factor1 = np.mean(1 + alpha * np.abs(F_arr[:, 1] - Phi))
    print(f"Mean rate factor: node 0 = {rate_factor0:.3f}, node 1 = {rate_factor1:.3f}")

    net.summary()

    # Example: print first 5 and last 5 F histories
    print("\nSample F history (first 5 events):")
    for k, F_vals in enumerate(net.history_F[:5]):
        print(f"  K={k}: F={['{:.3f}'.format(v) for v in F_vals]}")

    print("\nSample F history (last 5 events):")
    for offset, F_vals in enumerate(net.history_F[-5:]):
        k = net.K - 5 + offset
        print(f"  K={k}: F={['{:.3f}'.format(v) for v in F_vals]}")


if __name__ == "__main__":
    main()
