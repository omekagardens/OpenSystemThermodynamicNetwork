def example_two_node_basic():
    # 2 nodes: 0 = "emitter-ish", 1 = "accumulator-ish"
    net = OSTNNetwork(N=2, Phi_res=10.0, gamma=0.1, dt=1.0)

    # Initial free-levels
    net.set_initial_F([1.0, 5.0])

    # Different conductivities: node 0 a bit higher
    net.set_sigma([1.0, 0.3])

    # Gate fully open
    net.set_a([1.0, 1.0])

    # Simple constant physical power matrix P:
    # Node 0 sends to node 1; node 1 sends nothing.
    P = np.array([
        [0.0, 1.0],  # from 0 to 0,1
        [0.0, 0.0],  # from 1 to 0,1
    ])

    print("=== Example 1: two-node basic ===")
    print("tick, F0, F1, sigma0, sigma1")
    for k in range(10):
        F_new, G, G_res, R_tot = net.step(P)
        print(f"{k:02d}: {F_new[0]:6.3f}, {F_new[1]:6.3f}, "
              f"{net.sigma[0]:5.3f}, {net.sigma[1]:5.3f}")

if __name__ == "__main__":
    example_two_node_basic()
