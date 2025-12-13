from det_core import DETSystem, DETConfig
import time

class DETLab:
    @staticmethod
    def run_headless(sim: DETSystem, steps=100, log_freq=10, meta=None):
        """Standard runner for server/headless environments."""
        print(f"--- DET 2.0 EXPERIMENT START ---")
        print(f"Nodes: {sim.N} | Steps: {steps}")
        if meta is not None:
            print(f"Bridge endpoints: {meta.get('bridge_endpoints')}")
        if meta is None:
            print("-" * 60)
        if log_freq and log_freq > 0:
            print("Tick | E_total | Locked_total | Locked_A | Locked_B | I_bridge_B | Alive")
            print("-" * 86)
        
        history = []
        first_lock_B_tick = None
        bridge_cross_tick = None
        BRIDGE_I_THRESH = 0.05
        start_t = time.time()
        
        for i in range(steps):
            stats = sim.step()
            # Optional bridge/plant instrumentation
            if meta is not None:
                plantA_mask = meta["plantA_mask"].to(sim.F.device)
                plantB_mask = meta["plantB_mask"].to(sim.F.device)
                r1, r2 = meta["bridge_endpoints"]

                # Locked is defined by lateral gate being "closed"
                locked_vec = (sim.g_lateral < 0.5).float()
                locked_A = float((locked_vec * plantA_mask.float()).sum().item())
                locked_B = float((locked_vec * plantB_mask.float()).sum().item())

                # Track first tick Plant B locks any node
                if first_lock_B_tick is None and locked_B > 0:
                    # Use the loop index (i) so this matches the "Tick" column we print.
                    first_lock_B_tick = i

                # Signal at Plant B bridge endpoint:
                #   - raw can be signed (wave-like / oscillatory)
                #   - amp is rectified for a stable "amplitude" readout
                I_bridge_B_raw = float(sim.I[r2].item())
                I_bridge_B = max(I_bridge_B_raw, 0.0)

                # Track first meaningful bridge crossing (thresholded)
                if bridge_cross_tick is None and I_bridge_B >= BRIDGE_I_THRESH:
                    bridge_cross_tick = i

                # Stash into stats for downstream logging/analysis
                stats["locked_A"] = locked_A
                stats["locked_B"] = locked_B
                stats["I_bridge_B"] = I_bridge_B
                stats["first_lock_B_tick"] = first_lock_B_tick
                stats["bridge_cross_tick"] = bridge_cross_tick
                stats["I_bridge_B_raw"] = I_bridge_B_raw
            
            history.append(stats)
            
            if log_freq and log_freq > 0 and (i % log_freq == 0):
                # Be tolerant to either stats schema: (energy/locked/alive) or (total_energy/locked_nodes/alive_count)
                e = stats.get('energy', stats.get('total_energy', 0.0))
                locked_total = stats.get('locked', stats.get('locked_nodes', 0))
                alive = stats.get('alive', stats.get('alive_count', 0.0))

                if meta is None:
                    print(f"Tick {i:<4} | Energy: {e:.2f} | Locked: {locked_total} | Alive: {alive}")
                else:
                    I_amp = stats.get('I_bridge_B', 0.0)
                    I_raw = stats.get('I_bridge_B_raw', 0.0)
                    bc = stats.get('bridge_cross_tick', None)
                    bc_str = f"{bc}" if bc is not None else "-"
                    print(f"{i:<4} | {e:.2f} | {locked_total:<11} | {stats.get('locked_A',0):<8.0f} | {stats.get('locked_B',0):<8.0f} | {I_amp:<9.4f}({I_raw:>+9.4f}) | {alive} | bc={bc_str}")
                
        duration = time.time() - start_t
        print("-" * 60)
        print(f"Experiment Complete. Time: {duration:.4f}s")
        return history

    @staticmethod
    def inject_chaos_bite(sim: DETSystem, target_node, force=4.0):
        """API wrapper to inject trauma."""
        sim.F[target_node] += force