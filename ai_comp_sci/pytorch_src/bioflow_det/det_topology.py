import torch
import networkx as nx
import numpy as np
from typing import Dict, Tuple

class DETTopology:
    @staticmethod
    def create_organic_tree(num_nodes=20):
        """Returns adjacency matrices and root/leaf indices for a generic plant."""
        G = nx.barabasi_albert_graph(num_nodes, 1)
        pos = nx.spring_layout(G, center=(0,0), iterations=100)
        
        # Find Root (Lowest Y) and Leaf (Highest Y)
        y_vals = {i: pos[i][1] for i in range(num_nodes)}
        root = min(y_vals, key=y_vals.get)
        leaf = max(y_vals, key=y_vals.get)
        
        adj = torch.zeros(num_nodes, num_nodes)
        for u, v in G.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
            
        # Return E and I matrices (identical for single plant), plus metadata
        return adj, adj.clone(), root, leaf

    @staticmethod
    def create_fungal_bridge(nodes_per_plant=15):
        """Creates 2 plants connected by a high-conductivity info bridge."""
        total = nodes_per_plant * 2
        adj_E = torch.zeros(total, total)
        adj_I = torch.zeros(total, total)
        
        # Build Plant A (0 to N-1)
        adj_sub, _, r1, l1 = DETTopology.create_organic_tree(nodes_per_plant)
        adj_E[0:nodes_per_plant, 0:nodes_per_plant] = adj_sub
        adj_I[0:nodes_per_plant, 0:nodes_per_plant] = adj_sub
        
        # Build Plant B (N to 2N-1)
        adj_sub, _, r2_local, l2_local = DETTopology.create_organic_tree(nodes_per_plant)
        r2 = r2_local + nodes_per_plant
        l2 = l2_local + nodes_per_plant
        
        adj_E[nodes_per_plant:, nodes_per_plant:] = adj_sub
        adj_I[nodes_per_plant:, nodes_per_plant:] = adj_sub
        
        # Connect Roots via Fungal Bridge (Info Only)
        adj_I[r1, r2] = 5.0
        adj_I[r2, r1] = 5.0
        
        return adj_E, adj_I, (r1, r2), (l1, l2)

    @staticmethod
    def plant_masks(total: int, nodes_per_plant: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns boolean masks (plantA_mask, plantB_mask) for a 2-plant topology.
        """
        a_mask = torch.zeros(total, dtype=torch.bool)
        b_mask = torch.zeros(total, dtype=torch.bool)
        a_mask[:nodes_per_plant] = True
        b_mask[nodes_per_plant:2*nodes_per_plant] = True
        return a_mask, b_mask

    @staticmethod
    def bridge_edge_mask(total: int, r1: int, r2: int) -> torch.Tensor:
        """
        Returns a boolean NxN mask selecting the fungal bridge edges (r1<->r2).
        """
        m = torch.zeros(total, total, dtype=torch.bool)
        m[r1, r2] = True
        m[r2, r1] = True
        return m

    @staticmethod
    def bridge_metadata(nodes_per_plant: int, roots: Tuple[int, int], leaves: Tuple[int, int]) -> Dict:
        """
        Convenience metadata bundle for demos/loggers.
        """
        total = nodes_per_plant * 2
        r1, r2 = roots
        plantA_mask, plantB_mask = DETTopology.plant_masks(total, nodes_per_plant)
        bridge_mask = DETTopology.bridge_edge_mask(total, r1, r2)
        return {
            "nodes_per_plant": nodes_per_plant,
            "total": total,
            "roots": roots,
            "leaves": leaves,
            "plantA_mask": plantA_mask,
            "plantB_mask": plantB_mask,
            "bridge_mask": bridge_mask,
            "bridge_endpoints": (r1, r2),
        }