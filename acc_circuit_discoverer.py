import torch

from computational_graph import ComputationalGraph
from modular_addition_transformer import ModularAdditionTransformer

def build_computational_graph(model: ModularAdditionTransformer, granularity: str = "component") -> ComputationalGraph:
    """
    Build the computational graph for the modular addition transformer.

    granularity options:
    - "component": nodes are embed, attn_head_0..3, mlp, unembed
    - "fine": nodes include Q, K, V per head, individual MLP neuron groups, etc.
    - "neuron": individual MLP neurons as nodes
    """
    nodes = []
    edges = set()
    node_layers = {}

    if granularity == "component":
        # Layer 0: embeddings
        nodes.append("tok_embed")
        nodes.append("pos_embed")
        node_layers["tok_embed"] = 0
        node_layers["pos_embed"] = 0

        # Layer 1: attention heads
        for h in range(model.n_heads):
            name = f"attn_head_{h}"
            nodes.append(name)
            node_layers[name] = 1
            edges.add(("tok_embed", name))
            edges.add(("pos_embed", name))

        # Layer 2: MLP
        nodes.append("mlp")
        node_layers["mlp"] = 2
        edges.add(("tok_embed", "mlp"))
        edges.add(("pos_embed", "mlp"))
        for h in range(model.n_heads):
            edges.add((f"attn_head_{h}", "mlp"))

        # Layer 3: output (unembed)
        nodes.append("unembed")
        node_layers["unembed"] = 3
        edges.add(("mlp", "unembed"))
        # Residual stream: attention heads can also directly affect output
        for h in range(model.n_heads):
            edges.add((f"attn_head_{h}", "unembed"))
        # Embedding residual to output
        edges.add(("tok_embed", "unembed"))
        edges.add(("pos_embed", "unembed"))

    elif granularity == "fine":
        # Embeddings
        nodes.extend(["tok_embed", "pos_embed"])
        node_layers["tok_embed"] = 0
        node_layers["pos_embed"] = 0

        # Per-head Q, K, V
        for h in range(model.n_heads):
            for comp in ["Q", "K", "V"]:
                name = f"head_{h}_{comp}"
                nodes.append(name)
                node_layers[name] = 1
                edges.add(("tok_embed", name))
                edges.add(("pos_embed", name))

            # Head output
            head_out = f"attn_head_{h}"
            nodes.append(head_out)
            node_layers[head_out] = 2
            edges.add((f"head_{h}_Q", head_out))
            edges.add((f"head_{h}_K", head_out))
            edges.add((f"head_{h}_V", head_out))

        # MLP input (residual mid)
        nodes.append("residual_mid")
        node_layers["residual_mid"] = 3
        edges.add(("tok_embed", "residual_mid"))
        edges.add(("pos_embed", "residual_mid"))
        for h in range(model.n_heads):
            edges.add((f"attn_head_{h}", "residual_mid"))

        # MLP
        nodes.append("mlp_pre")
        node_layers["mlp_pre"] = 4
        edges.add(("residual_mid", "mlp_pre"))

        nodes.append("mlp_hidden")
        node_layers["mlp_hidden"] = 5
        edges.add(("mlp_pre", "mlp_hidden"))

        nodes.append("mlp_out")
        node_layers["mlp_out"] = 6
        edges.add(("mlp_hidden", "mlp_out"))

        # Output
        nodes.append("unembed")
        node_layers["unembed"] = 7
        edges.add(("residual_mid", "unembed"))
        edges.add(("mlp_out", "unembed"))

    elif granularity == "neuron":
        # Embeddings
        nodes.extend(["tok_embed", "pos_embed"])
        node_layers["tok_embed"] = 0
        node_layers["pos_embed"] = 0

        # Attention heads
        for h in range(model.n_heads):
            name = f"attn_head_{h}"
            nodes.append(name)
            node_layers[name] = 1
            edges.add(("tok_embed", name))
            edges.add(("pos_embed", name))

        # Individual MLP neurons (grouped into clusters for tractability)
        n_neuron_groups = min(model.d_mlp, 64)  # Group neurons for tractability
        neurons_per_group = model.d_mlp // n_neuron_groups
        for g in range(n_neuron_groups):
            name = f"neuron_group_{g}"
            nodes.append(name)
            node_layers[name] = 2
            edges.add(("tok_embed", name))
            edges.add(("pos_embed", name))
            for h in range(model.n_heads):
                edges.add((f"attn_head_{h}", name))

        # Output
        nodes.append("unembed")
        node_layers["unembed"] = 3
        for g in range(n_neuron_groups):
            edges.add((f"neuron_group_{g}", "unembed"))
        for h in range(model.n_heads):
            edges.add((f"attn_head_{h}", "unembed"))
        edges.add(("tok_embed", "unembed"))
        edges.add(("pos_embed", "unembed"))

    return ComputationalGraph(nodes=nodes, edges=edges, node_layers=node_layers)



class ACDCCircuitDiscoverer:
    """
    Implements the ACDC algorithm from Conmy et al. (2023).

    The algorithm iterates from outputs to inputs through the computational graph,
    starting at the output node, to build a subgraph. At every node it attempts to
    remove as many edges that enter this node as possible, without reducing the
    model's performance on a selected metric (KL divergence).
    """

    def __init__(self, model: ModularAdditionTransformer, granularity: str = "component"):
        self.model = model
        self.P = model.P
        self.granularity = granularity
        self.graph = build_computational_graph(model, granularity)

        # Cache clean and corrupted activations
        self._clean_cache = None
        self._corrupted_cache = None
        self._clean_logits = None
        self._clean_a = None
        self._clean_b = None
        self._corrupt_a = None
        self._corrupt_b = None

    def _prepare_data(self, n_samples: int = 512):
        """Prepare clean and corrupted datasets for activation patching."""
        P = self.P
        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)

        # Use a subset for efficiency
        if n_samples < P * P:
            perm = torch.randperm(P * P)[:n_samples]
            clean_a = all_a[perm]
            clean_b = all_b[perm]
        else:
            clean_a = all_a
            clean_b = all_b

        # Corrupted data: random permutation (interchange intervention)
        corrupt_perm = torch.randperm(len(clean_a))
        corrupt_a = clean_a[corrupt_perm]
        corrupt_b = clean_b[corrupt_perm]

        self._clean_a = clean_a
        self._clean_b = clean_b
        self._corrupt_a = corrupt_a
        self._corrupt_b = corrupt_b

        # Cache activations
        with torch.no_grad():
            self._clean_logits, self._clean_cache = self.model.forward_with_hooks(clean_a, clean_b)
            _, self._corrupted_cache = self.model.forward_with_hooks(corrupt_a, corrupt_b)

    def _compute_kl_divergence(self, logits_patched: torch.Tensor) -> float:
        """
        Compute KL divergence between clean model output and patched output.
        DKL(G(x) || H(x, x'))
        """
        clean_probs = F.softmax(self._clean_logits, dim=-1)
        patched_log_probs = F.log_softmax(logits_patched, dim=-1)

        # KL(clean || patched) = sum(clean * (log(clean) - log(patched)))
        kl = F.kl_div(patched_log_probs, clean_probs, reduction="batchmean")
        return kl.item()

    def _get_patch_for_edge(self, parent: str, child: str) -> dict:
        """
        Create a patch dict that replaces the contribution of parent to child
        with the corrupted activation.
        """
        patches = {}

        if self.granularity == "component":
            # For component-level, patching an edge means replacing the parent's
            # output with its corrupted version when it flows to the child.
            # We map edge types to the appropriate activation to patch.

            if parent == "tok_embed":
                patches["tok_embed"] = self._corrupted_cache["tok_embed"]
            elif parent == "pos_embed":
                patches["pos_embed"] = self._corrupted_cache["pos_embed"]
            elif parent.startswith("attn_head_"):
                head_idx = int(parent.split("_")[-1])
                patches[f"attn_head_{head_idx}"] = self._corrupted_cache[f"attn_head_{head_idx}"]
            elif parent == "mlp":
                patches["mlp_out"] = self._corrupted_cache["mlp_out"]

        elif self.granularity == "fine":
            if parent in self._corrupted_cache:
                patches[parent] = self._corrupted_cache[parent]

        elif self.granularity == "neuron":
            if parent.startswith("neuron_group_"):
                # Patch specific neuron group by zeroing/corrupting those neurons
                group_idx = int(parent.split("_")[-1])
                n_groups = min(self.model.d_mlp, 64)
                neurons_per_group = self.model.d_mlp // n_groups
                start = group_idx * neurons_per_group
                end = start + neurons_per_group

                corrupted_hidden = self._corrupted_cache["mlp_hidden"].clone()
                clean_hidden = self._clean_cache["mlp_hidden"].clone()
                # Create a patched version where only this group is corrupted
                patched_hidden = clean_hidden.clone()
                patched_hidden[:, :, start:end] = corrupted_hidden[:, :, start:end]
                patches["mlp_hidden"] = patched_hidden
            elif parent in self._corrupted_cache:
                patches[parent] = self._corrupted_cache[parent]

        return patches

    def _evaluate_edge_importance(self, parent: str, child: str) -> float:
        """
        Evaluate the importance of an edge by patching it and measuring KL divergence.
        Following Algorithm 1 from Conmy et al. (2023): we measure
        DKL(G(x) || H_new(x, x')) - DKL(G(x) || H(x, x'))
        """
        patches = self._get_patch_for_edge(parent, child)
        if not patches:
            return 0.0

        with torch.no_grad():
            patched_logits = self.model.forward_with_patches(
                self._clean_a, self._clean_b, patches
            )

        kl_div = self._compute_kl_divergence(patched_logits)
        return kl_div

    def run_acdc(self, threshold: float = 0.01, n_samples: int = 512,
                 progress_cb=None) -> ComputationalGraph:

        """
        Run the full ACDC algorithm to discover the minimal circuit.

        Following Algorithm 1 from Conmy et al. (2023): iterates from outputs to inputs
        through the computational graph, attempting to remove edges that don't significantly
        affect model performance (measured by KL divergence).

        Args:
            threshold: tau parameter - edges causing KL increase below this are removed
            n_samples: number of data samples for activation patching
            progress_cb: callback for progress messages

        Returns:
            Pruned ComputationalGraph representing the discovered circuit
        """
        def update(msg):
            if progress_cb:
                progress_cb(msg)

        update("Preparing clean and corrupted datasets...")
        self._prepare_data(n_samples)

        # Initialize H to the full computational graph (Algorithm 1, Line 1)
        H = self.graph.copy()
        initial_edges = H.num_edges

        update(f"Starting ACDC with {initial_edges} edges, threshold tau={threshold}")

        # Compute baseline KL divergence (should be 0 for full graph)
        baseline_kl = self._compute_circuit_kl(H)
        update(f"Baseline KL divergence (full graph): {baseline_kl:.6f}")

        # Sort nodes from output to input (reverse topological order) - Algorithm 1, Line 2
        sorted_nodes = H.reverse_topological_sort()

        edges_removed = 0
        edges_tested = 0

        # Iterate over nodes from output to input (Algorithm 1, Line 3)
        for node_idx, v in enumerate(sorted_nodes):
            parents = H.get_parents(v)
            if not parents:
                continue

            update(f"Processing node '{v}' ({node_idx+1}/{len(sorted_nodes)}) - {len(parents)} parent edges to test")

            # For each parent w of v (Algorithm 1, Line 4)
            for w in list(parents):  # copy list since we may modify edges
                edges_tested += 1

                # Temporarily remove candidate edge (Algorithm 1, Line 5)
                H_new = H.copy()
                H_new.remove_edge(w, v)

                # Compute KL divergence change (Algorithm 1, Line 6)
                kl_new = self._compute_circuit_kl(H_new)
                kl_current = self._compute_circuit_kl(H)
                kl_increase = kl_new - kl_current

                # If edge is unimportant, remove permanently (Algorithm 1, Lines 6-7)
                if kl_increase < threshold:
                    H.remove_edge(w, v)
                    edges_removed += 1

            remaining = H.num_edges
            update(f"  After node '{v}': {remaining} edges remaining ({edges_removed} removed so far)")

        final_edges = H.num_edges
        update(f"\nACDC complete: {initial_edges} -> {final_edges} edges "
               f"({edges_removed} removed, {edges_tested} tested)")
        update(f"Compression ratio: {final_edges/initial_edges:.2%}")

        return H

    def _compute_circuit_kl(self, H: ComputationalGraph) -> float:
        """
        Compute KL divergence for a given subgraph H.

        For edges NOT in H, we replace their activations with corrupted activations.
        This implements H(x_i, x'_i) from the ACDC paper: the model output when
        edges not in H are overwritten with their corrupted values.
        """
        # Determine which edges are NOT in H (these get patched with corrupted activations)
        full_edges = self.graph.edges
        removed_edges = full_edges - H.edges

        if not removed_edges:
            # Full graph, no patching needed - KL should be ~0
            return 0.0

        # Build patches dict based on removed edges
        patches = self._build_patches_from_removed_edges(removed_edges)

        # Run forward pass with patches
        with torch.no_grad():
            patched_logits = self.model.forward_with_patches(
                self._clean_a, self._clean_b, patches
            )

        kl_div = self._compute_kl_divergence(patched_logits)
        return kl_div

    def _build_patches_from_removed_edges(self, removed_edges: set) -> dict:
        """
        Build a patches dictionary from a set of removed edges.

        When an edge (parent -> child) is removed, we replace the parent's
        contribution to the child with the corrupted activation value.

        For the modular addition transformer, we implement this at the component level
        by determining which activations need to be replaced.
        """
        patches = {}

        # Count how many edges into each node are removed
        # If ALL edges into a node are removed, we patch that node's activation entirely
        node_incoming_total = {}
        node_incoming_removed = {}

        for parent, child in self.graph.edges:
            node_incoming_total[child] = node_incoming_total.get(child, 0) + 1

        for parent, child in removed_edges:
            node_incoming_removed[child] = node_incoming_removed.get(child, 0) + 1

        if self.granularity == "component":
            # Component-level patching strategy:
            # If all inputs to a component are corrupted, replace its output with corrupted version

            for node, n_removed in node_incoming_removed.items():
                n_total = node_incoming_total.get(node, 0)

                if n_total > 0 and n_removed == n_total:
                    # All inputs removed - fully patch this node
                    if node in self._corrupted_cache:
                        patches[node] = self._corrupted_cache[node]
                elif n_removed > 0 and n_removed < n_total:
                    # Partial patching - interpolate based on fraction removed
                    # This is an approximation; true edge-level patching would require
                    # decomposing the residual stream contributions
                    if node in self._corrupted_cache and node in self._clean_cache:
                        frac_removed = n_removed / n_total
                        clean_act = self._clean_cache[node]
                        corrupt_act = self._corrupted_cache[node]
                        patches[node] = (1 - frac_removed) * clean_act + frac_removed * corrupt_act

            # Handle specific edge patterns for more precise patching
            for parent, child in removed_edges:
                # Attention head outputs directly to unembed (residual stream bypass)
                if parent.startswith("attn_head_") and child == "unembed":
                    head_idx = int(parent.split("_")[-1])
                    patch_key = f"attn_head_{head_idx}"
                    if patch_key in self._corrupted_cache and patch_key not in patches:
                        patches[patch_key] = self._corrupted_cache[patch_key]

                # Embedding to attention head
                if parent in ("tok_embed", "pos_embed") and child.startswith("attn_head_"):
                    if parent in self._corrupted_cache and parent not in patches:
                        patches[parent] = self._corrupted_cache[parent]

        elif self.granularity == "fine":
            # Fine-grained patching: directly patch specific intermediate activations
            for parent, child in removed_edges:
                if parent in self._corrupted_cache and parent not in patches:
                    # Check if ALL edges from this parent are removed
                    parent_children = [(p, c) for p, c in self.graph.edges if p == parent]
                    parent_removed = [(p, c) for p, c in removed_edges if p == parent]
                    if len(parent_removed) == len(parent_children):
                        patches[parent] = self._corrupted_cache[parent]

        elif self.granularity == "neuron":
            # Neuron-group level patching
            for parent, child in removed_edges:
                if parent.startswith("neuron_group_"):
                    group_idx = int(parent.split("_")[-1])
                    # Patch the corresponding neurons in mlp_hidden
                    if "mlp_hidden" not in patches:
                        patches["mlp_hidden"] = self._clean_cache["mlp_hidden"].clone()
                    neurons_per_group = self.model.d_mlp // 64
                    start = group_idx * neurons_per_group
                    end = start + neurons_per_group
                    patches["mlp_hidden"][:, :, start:end] = \
                        self._corrupted_cache["mlp_hidden"][:, :, start:end]

                elif parent.startswith("attn_head_") and child == "unembed":
                    head_idx = int(parent.split("_")[-1])
                    patch_key = f"attn_head_{head_idx}"
                    if patch_key in self._corrupted_cache:
                        patches[patch_key] = self._corrupted_cache[patch_key]

        return patches

    def get_circuit_summary(self, circuit: ComputationalGraph) -> dict:
        """
        Summarize the discovered circuit: which components are included,
        which edges remain, and what the circuit structure looks like.
        """
        # Find active nodes (nodes with at least one edge)
        active_nodes = set()
        for parent, child in circuit.edges:
            active_nodes.add(parent)
            active_nodes.add(child)

        # Categorize edges by type
        edge_types = {
            "embed_to_attn": [],
            "embed_to_mlp": [],
            "attn_to_mlp": [],
            "attn_to_output": [],
            "mlp_to_output": [],
            "embed_to_output": [],
            "other": [],
        }

        for parent, child in circuit.edges:
            if parent in ("tok_embed", "pos_embed") and child.startswith("attn"):
                edge_types["embed_to_attn"].append((parent, child))
            elif parent in ("tok_embed", "pos_embed") and child == "mlp":
                edge_types["embed_to_mlp"].append((parent, child))
            elif parent.startswith("attn") and child == "mlp":
                edge_types["attn_to_mlp"].append((parent, child))
            elif parent.startswith("attn") and child == "unembed":
                edge_types["attn_to_output"].append((parent, child))
            elif parent == "mlp" and child == "unembed":
                edge_types["mlp_to_output"].append((parent, child))
            elif parent in ("tok_embed", "pos_embed") and child == "unembed":
                edge_types["embed_to_output"].append((parent, child))
            else:
                edge_types["other"].append((parent, child))

        # Identify which attention heads are in the circuit
        active_heads = [n for n in active_nodes if n.startswith("attn_head_")]

        return {
            "num_edges": circuit.num_edges,
            "num_active_nodes": len(active_nodes),
            "active_nodes": sorted(active_nodes),
            "active_heads": sorted(active_heads),
            "edge_types": {k: len(v) for k, v in edge_types.items()},
            "edge_details": edge_types,
            "has_mlp_path": any(child == "unembed" for _, child in circuit.edges
                               if _ == "mlp"),
            "has_direct_path": any(parent in ("tok_embed", "pos_embed")
                                   for parent, child in circuit.edges if child == "unembed"),
        }

    def visualize_circuit(self, circuit: ComputationalGraph) -> str:
        """
        Create a text-based visualization of the discovered circuit.
        """
        summary = self.get_circuit_summary(circuit)

        lines = []
        lines.append("=" * 60)
        lines.append("ACDC Discovered Circuit")
        lines.append("=" * 60)
        lines.append(f"Total edges: {summary['num_edges']} / {self.graph.num_edges}")
        lines.append(f"Active nodes: {summary['num_active_nodes']} / {len(self.graph.nodes)}")
        lines.append(f"Compression: {summary['num_edges']/self.graph.num_edges:.1%}")
        lines.append("")
        lines.append("Active components:")
        for node in summary["active_nodes"]:
            lines.append(f"  - {node}")
        lines.append("")
        lines.append("Edge breakdown:")
        for edge_type, count in summary["edge_types"].items():
            if count > 0:
                lines.append(f"  {edge_type}: {count}")
        lines.append("")
        lines.append("Circuit structure:")
        lines.append("  Input -> Attention -> MLP -> Output")
        if summary["has_mlp_path"]:
            lines.append("  [MLP pathway ACTIVE]")
        if summary["has_direct_path"]:
            lines.append("  [Direct residual pathway ACTIVE]")
        if summary["active_heads"]:
            lines.append(f"  Active attention heads: {summary['active_heads']}")

        return "\n".join(lines)


