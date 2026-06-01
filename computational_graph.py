from dataclasses import dataclass, field

@dataclass
class ComputationalGraph:
    """
    Represents the computational graph of the transformer at a chosen granularity.
    Nodes represent components (embed, attn heads, MLP neurons, unembed).
    Edges represent information flow between components.
    """
    nodes: list  # list of node names
    edges: set   # set of (parent, child) tuples
    node_layers: dict  # node_name -> layer_index for topological ordering

    def reverse_topological_sort(self) -> list:
        """Sort nodes from output to input (reverse topological order)."""
        return sorted(self.nodes, key=lambda n: -self.node_layers.get(n, 0))

    def remove_edge(self, parent: str, child: str):
        """Remove an edge from the graph."""
        self.edges.discard((parent, child))

    def get_parents(self, node: str) -> list:
        """Get all parent nodes of a given node."""
        return [p for p, c in self.edges if c == node]

    def get_children(self, node: str) -> list:
        """Get all child nodes of a given node."""
        return [c for p, c in self.edges if p == node]

    def copy(self):
        """Return a deep copy of this graph."""
        return ComputationalGraph(
            nodes=list(self.nodes),
            edges=set(self.edges),
            node_layers=dict(self.node_layers),
        )

    @property
    def num_edges(self) -> int:
        return len(self.edges)
