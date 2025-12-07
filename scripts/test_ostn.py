import numpy as np

class OSTNNetwork:
    """
    Simple Open-System Thermodynamic Network (OSTN) simulator
    implementing the DET 2.0 tick equations.
    """

    def __init__(self, N, Phi_res=10.0, gamma=0.1, dt=1.0,
                 alpha_E=1.0, alpha_I=0.0, alpha_T=0.0):
        """
        Parameters
        ----------
        N : int
            Number of active nodes (excluding reservoir).
        Phi_res : float
            Reservoir potential Phi_res.
        gamma : float
            Cost coefficient for outgoing flow.
        dt : float
            Tick duration Delta t.
        alpha_E, alpha_I, alpha_T : float
            Weights for physical, informational, and activity components.
        """
        self.N = N
        self.Phi_res = Phi_res
        self.gamma = gamma
        self.dt = dt
        self.alpha_E = alpha_E
        self.alpha_I = alpha_I
        self.alpha_T = alpha_T

        # State variables
        self.F = np.zeros(N, dtype=float)          # free-levels F_i
        self.sigma = np.ones(N, dtype=float)       # conductivities sigma_i
        self.a = np.ones(N, dtype=float)           # gating factors a_i in [0,1]

        # Transfer efficiencies eta_{j->i}, default 1 for j != i, 0 on diagonal
        self.eta = np.ones((N, N), dtype=float)
        np.fill_diagonal(self.eta, 0.0)

        # Adaptation parameters
        self.epsilon = np.zeros(N, dtype=float)    # efficiencies epsilon_i
        self.eta_sigma = 0.0                       # learning rate (0 => disabled)
        self.eps_denom = 1e-9                      # small epsilon to avoid /0

    # --- Configuration helpers -------------------------------------------------

    def set_initial_F(self, F0):
        self.F = np.array(F0, dtype=float)

    def set_sigma(self, sigma):
        self.sigma = np.array(sigma, dtype=float)

    def set_a(self, a):
        self.a = np.array(a, dtype=float)

    def set_eta(self, eta_matrix):
        self.eta = np.array(eta_matrix, dtype=float)

    def set_adaptation(self, eta_sigma=0.0, eps_denom=1e-9):
        """Enable/disable conductivity adaptation."""
        self.eta_sigma = float(eta_sigma)
        self.eps_denom = float(eps_denom)

    # --- Core math -------------------------------------------------------------

    def _compute_G_matrix(self, P, I_rate=None, A=None):
        """
        Compute G_{i->j}^{(k)} for the current tick from P, I_rate, A matrices.
        All matrices are N x N with P[i,j] meaning flow i->j.
        """
        P = np.asarray(P, dtype=float)
        if I_rate is None:
            I_rate = np.zeros_like(P)
        else:
            I_rate = np.asarray(I_rate, dtype=float)
        if A is None:
            A = np.zeros_like(P)
        else:
            A = np.asarray(A, dtype=float)

        J = (self.alpha_E * P +
             self.alpha_I * I_rate +
             self.alpha_T * A)
        G = self.dt * J
        return G

    def step(self, P, I_rate=None, A=None, adapt_sigma=False, f=None):
        """
        Advance the network by one tick.

        Parameters
        ----------
        P, I_rate, A : array_like, shape (N, N)
            Components of J_{i->j}. Any of I_rate or A may be None.
        adapt_sigma : bool
            If True, update sigma_i according to epsilon_i and f().
        f : callable or None
            Nonlinearity for conductivity adaptation. If None and
            adapt_sigma is True, uses a default tanh-like function.

        Returns
        -------
        F_new : np.ndarray
            Updated free-levels after this tick.
        G : np.ndarray
            Inter-node tick flows G_{i->j}^{(k)}.
        G_res : np.ndarray
            Reservoir tick inflows G_i^{res,(k)}.
        R_tot : np.ndarray
            Total incoming (network + reservoir) R_i^{tot,(k)}.
        """
        N = self.N

        # 1. Inter-node flows
        G = self._compute_G_matrix(P, I_rate, A)      # G_{i->j}
        G_out = G.sum(axis=1)                         # G_i^{out}
        R = G.sum(axis=0)                             # R_i (incoming from other nodes)

        # 2. Reservoir coupling
        grad = np.maximum(0.0, self.Phi_res - self.F)  # max(0, Phi_res - F_i)
        G_res = self.a * self.sigma * grad * self.dt   # G_i^{res}
        R_tot = R + G_res                              # R_i^{tot}

        # 3. Free-level update
        #    F_i^{(k+1)}
        #      = F_i^{(k)} - gamma G_i^{out}
        #        + sum_j eta_{j->i} G_{j->i} + G_i^{res}
        F_new = (self.F
                 - self.gamma * G_out
                 + (self.eta * G).sum(axis=0)  # sum_j eta_{j->i} G_{j->i}
                 + G_res)

        # 4. Optional conductivity adaptation
        if adapt_sigma and self.eta_sigma != 0.0:
            self.epsilon = R_tot / (G_out + self.eps_denom)
            if f is None:
                # Smooth bounded response in [-1, 1], centered around epsilon ~ 1
                def f(x):
                    return np.tanh(x - 1.0)
            delta_sigma = self.eta_sigma * f(self.epsilon)
            self.sigma = np.clip(self.sigma + delta_sigma, 0.0, None)

        # commit new state
        self.F = F_new
        return F_new.copy(), G, G_res, R_tot
