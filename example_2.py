def example_with_adaptation():
    net = OSTNNetwork(N=2, Phi_res=10.0, gamma=0.1, dt=1.0)

    # Start both at same potential
    net.set_initial_F([1.0, 1.0])

    # Slightly different starting sigmas
    net.set_sigma([0.5, 0.5])
    net.set_a([1.0, 1.0])

    # Fully efficient transfer eta_{j->i} = 1 for j != i
    net.set_eta([[0.0, 1.0],
                 [1.0, 0.0]])

    # Enable adaptation
    net.set_adaptation(eta_sigma=0.05, eps_denom=1e-9)

    # Constant export from node 0 to 1
    P = np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ])

    # Simple adaptation nonlinearity: up-regulate when epsilon > 1,
    # down-regulate when epsilon < 1
    def adapt_fn(eps):
        return np.tanh(eps - 1.0)

    print("\n=== Example 2: with conductivity adaptation ===")
    print("tick, F0, F1, sigma0, sigma1, eps0, eps1")
    for k in range(20):
        F_new, G, G_res, R_tot = net.step(
            P,
            adapt_sigma=True,
            f=adapt_fn,
        )
        eps = net.epsilon
        print(
            f"{k:02d}: "
            f"F0={F_new[0]:6.3f}, F1={F_new[1]:6.3f}, "
            f"s0={net.sigma[0]:5.3f}, s1={net.sigma[1]:5.3f}, "
            f"e0={eps[0]:5.3f}, e1={eps[1]:5.3f}"
        )

if __name__ == "__main__":
    example_two_node_basic()
    example_with_adaptation()
