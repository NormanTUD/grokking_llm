# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "numpy",
#     "scipy",
#     "plotly",
#     "gradio",
#     "pandas",
# ]
# ///
"""
Grokking Circuit Discovery & Verification Tool

Automatically discovers the Fourier multiplication algorithm (and similar
structured circuits) in small transformers trained on modular arithmetic,
based on Nanda et al. (2023) "Progress Measures for Grokking via
Mechanistic Interpretability".

Enhanced with Automatic Circuit Discovery (ACDC) methods from
Conmy et al. (2023) "Towards Automated Circuit Discovery for
Mechanistic Interpretability".

The tool:
1. Trains (or loads) a small transformer on modular addition
2. Discovers key frequencies via Fourier analysis of weights
3. Performs ACDC-style edge-level activation patching to find minimal circuits
4. Verifies the circuit mechanically (ablations, FVE checks)
5. Visualizes embeddings on the circle, neuron activations, attention patterns
6. Provides mathematical descriptions of discovered circuits
7. Tests predictions exhaustively to confirm correctness

Usage:
    uv run circuit_extract.py
"""


import sys
import os
import shutil
import subprocess
import time
from pathlib import Path

# =============================================================================
# Auto-restart under `uv run` if invoked directly with python3
# =============================================================================

def _ensure_uv_run():
    """
    Detect if this script was invoked directly (e.g. `python3 grok.py`) rather
    than via `uv run grok.py`. If so, attempt to re-exec under `uv run` with
    all original arguments. If `uv` is not installed, print instructions and exit.
    """
    # If UV_RUN is set in the environment, we're already inside `uv run`
    # uv sets several env vars when running; we check for the virtual env it creates
    # A reliable heuristic: check if we're inside a uv-managed venv or if the
    # parent process is uv. We use a custom env var approach for certainty.
    if os.environ.get("_UV_RUN_ACTIVE") == "1":
        return  # Already running under uv run, proceed normally

    # We're NOT running under uv run — attempt to re-exec
    uv_path = shutil.which("uv")

    if uv_path is None:
        print("=" * 60)
        print("ERROR: This script must be run with `uv run` but `uv` was")
        print("not found on your system.")
        print("=" * 60)
        print()
        print("To install uv, run one of the following:")
        print()
        print("  # On macOS/Linux:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print()
        print("  # On Windows:")
        print("  powershell -ExecutionPolicy ByPass -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
        print()
        print("  # Or via pip (not recommended):")
        print("  pip install uv")
        print()
        print("Once installed, run this script with:")
        print(f"  uv run {os.path.basename(__file__)}")
        print()
        print("=" * 60)
        sys.exit(1)

    # uv is available — re-exec this script under `uv run`
    script_path = os.path.abspath(__file__)
    # Pass along any extra CLI arguments the user may have provided
    extra_args = sys.argv[1:]

    cmd = [uv_path, "run", script_path] + extra_args

    print(f"[auto-restart] Detected direct invocation (python3 {os.path.basename(__file__)})")
    print(f"[auto-restart] Re-launching with: {' '.join(cmd)}")
    print()

    # Set the marker env var so the re-launched process knows it's under uv
    env = os.environ.copy()
    env["_UV_RUN_ACTIVE"] = "1"

    # Replace the current process with uv run
    if sys.platform == "win32":
        # On Windows, os.execvpe may not work reliably; use subprocess instead
        result = subprocess.run(cmd, env=env)
        sys.exit(result.returncode)
    else:
        os.execvpe(uv_path, cmd, env)


# Run the check immediately at import time, before any heavy imports
_ensure_uv_run()

import warnings
import math
import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Directory to save training logs
SAVE_DIR = "training_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# LaTeX Equation Generator for Discovered Circuits
# =============================================================================

from circuit_latex_generator import CircuitLatexGenerator

# =============================================================================
# Minimal Transformer for Modular Arithmetic
# =============================================================================

from modular_addition_transformer import ModularAdditionTransformer

def find_circular_dimension_pairs(model: ModularAdditionTransformer,
                                   key_frequencies: list[int] = None,
                                   top_k_pairs: int = 5) -> list[dict]:
    """
    Automatically find dimension pairs in the embedding space that show
    circular structure (i.e., tokens arranged on a circle).

    Strategy:
    1. If key_frequencies are known, project embeddings onto the Fourier basis
       and find the (cos_k, sin_k) direction pairs — these are guaranteed circles.
    2. Also do a brute-force search using circularity score (variance of radii)
       over all dimension pairs, but only report the top-k.

    Returns a list of dicts: [{dim_x, dim_y, frequency, circularity_score, description}, ...]
    """
    P = model.P
    d_model = model.d_model
    W_E = model.embed.weight[:P].detach().cpu().numpy()  # (P, d_model)

    results = []

    # === Method 1: Fourier-basis projection (best if key_frequencies known) ===
    if key_frequencies:
        # Build Fourier basis
        fourier_basis = np.zeros((P, P))
        fourier_basis[0] = np.ones(P) / np.sqrt(P)
        for k in range(1, P // 2 + 1):
            fourier_basis[2*k - 1] = np.cos(2 * np.pi * k * np.arange(P) / P) * np.sqrt(2/P)
            if 2*k < P:
                fourier_basis[2*k] = np.sin(2 * np.pi * k * np.arange(P) / P) * np.sqrt(2/P)

        # Project embedding onto Fourier basis: (P, P) @ (P, d_model) -> (P, d_model)
        # Each row of W_E_fourier corresponds to a Fourier component
        W_E_fourier = fourier_basis @ W_E  # (P, d_model)

        for k in key_frequencies:
            # The cos and sin rows for frequency k
            cos_row = W_E_fourier[2*k - 1]  # (d_model,) — direction of cos(2πk·t/P)
            sin_row = W_E_fourier[2*k] if 2*k < P else np.zeros(d_model)  # direction of sin(2πk·t/P)

            # Project all token embeddings onto these two directions
            # This gives us the "natural" 2D plane for frequency k
            x_proj = W_E @ cos_row / (np.linalg.norm(cos_row)**2 + 1e-10)  # (P,)
            y_proj = W_E @ sin_row / (np.linalg.norm(sin_row)**2 + 1e-10)  # (P,)

            # Compute circularity score
            cx, cy = x_proj.mean(), y_proj.mean()
            radii = np.sqrt((x_proj - cx)**2 + (y_proj - cy)**2)
            mean_r = radii.mean()
            circularity = 1.0 - (radii.std() / (mean_r + 1e-10))  # 1.0 = perfect circle

            results.append({
                "dim_x": f"fourier_cos_{k}",
                "dim_y": f"fourier_sin_{k}",
                "frequency": k,
                "circularity_score": float(circularity),
                "x_coords": x_proj,
                "y_coords": y_proj,
                "description": f"Frequency k={k}: Fourier projection (cos_{k}, sin_{k})",
                "method": "fourier",
            })

    # === Method 2: PCA on embedding to find top circular planes ===
    # Use SVD to find principal components, then check pairs for circularity
    from scipy.linalg import svd
    W_centered = W_E - W_E.mean(axis=0, keepdims=True)
    U, S, Vt = svd(W_centered, full_matrices=False)

    # Check top principal component pairs for circularity
    n_components_to_check = min(20, d_model)

    pair_scores = []
    for i in range(n_components_to_check):
        for j in range(i+1, n_components_to_check):
            x_proj = U[:, i] * S[i]
            y_proj = U[:, j] * S[j]

            cx, cy = x_proj.mean(), y_proj.mean()
            radii = np.sqrt((x_proj - cx)**2 + (y_proj - cy)**2)
            mean_r = radii.mean()
            if mean_r < 1e-10:
                continue
            circularity = 1.0 - (radii.std() / mean_r)

            # Also check if tokens are evenly spaced (angular uniformity)
            angles = np.arctan2(y_proj - cy, x_proj - cx)
            sorted_angles = np.sort(angles)
            angle_diffs = np.diff(sorted_angles)
            angular_uniformity = 1.0 - (angle_diffs.std() / (angle_diffs.mean() + 1e-10))

            combined_score = 0.7 * circularity + 0.3 * angular_uniformity

            pair_scores.append({
                "dim_x": f"PC_{i}",
                "dim_y": f"PC_{j}",
                "pc_i": i,
                "pc_j": j,
                "frequency": None,  # Unknown — could detect by counting cycles
                "circularity_score": float(combined_score),
                "x_coords": x_proj,
                "y_coords": y_proj,
                "description": f"PCA pair (PC{i}, PC{j}): circularity={combined_score:.3f}",
                "method": "pca",
            })

    # Sort by score and take top-k
    pair_scores.sort(key=lambda x: -x["circularity_score"])
    results.extend(pair_scores[:top_k_pairs])

    # === Method 3: Raw dimension pairs (fast scan) ===
    # Only check if d_model is small enough, or sample randomly
    if d_model <= 32:
        raw_pairs = [(i, j) for i in range(d_model) for j in range(i+1, d_model)]
    else:
        # Sample random pairs + pairs near high-variance dimensions
        variances = W_E.var(axis=0)
        top_dims = np.argsort(variances)[-20:]  # Top 20 highest-variance dims
        raw_pairs = [(int(top_dims[i]), int(top_dims[j]))
                     for i in range(len(top_dims)) for j in range(i+1, len(top_dims))]

    raw_scores = []
    for (i, j) in raw_pairs:
        x_proj = W_E[:, i]
        y_proj = W_E[:, j]
        cx, cy = x_proj.mean(), y_proj.mean()
        radii = np.sqrt((x_proj - cx)**2 + (y_proj - cy)**2)
        mean_r = radii.mean()
        if mean_r < 1e-10:
            continue
        circularity = 1.0 - (radii.std() / mean_r)
        raw_scores.append({
            "dim_x": i,
            "dim_y": j,
            "frequency": None,
            "circularity_score": float(circularity),
            "x_coords": x_proj,
            "y_coords": y_proj,
            "description": f"Raw dims ({i}, {j}): circularity={circularity:.3f}",
            "method": "raw",
        })

    raw_scores.sort(key=lambda x: -x["circularity_score"])
    results.extend(raw_scores[:top_k_pairs])

    # Sort all results by circularity score
    results.sort(key=lambda x: -x["circularity_score"])

    return results


def make_auto_circle_plot(model: ModularAdditionTransformer,
                          pair_info: dict) -> go.Figure:
    """
    Create a circle plot from a pre-computed pair_info dict
    (as returned by find_circular_dimension_pairs).
    """
    P = model.P
    x_coords = pair_info["x_coords"]
    y_coords = pair_info["y_coords"]

    fig = go.Figure()

    # Color by token ID to show ordering
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers+text",
        marker=dict(size=10, color=np.arange(P), colorscale="hsv", showscale=True,
                    colorbar=dict(title="Token ID")),
        text=[str(i) for i in range(P)],
        textposition="top center",
        textfont=dict(size=7),
        hovertemplate=(
            "Token: %{text}<br>"
            "x: %{x:.4f}<br>"
            "y: %{y:.4f}<br>"
            "<extra></extra>"
        ),
        name="Tokens",
    ))

    # Fit circle
    cx, cy = x_coords.mean(), y_coords.mean()
    radii = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    avg_r = radii.mean()

    theta = np.linspace(0, 2*np.pi, 200)
    fig.add_trace(go.Scatter(
        x=cx + avg_r * np.cos(theta),
        y=cy + avg_r * np.sin(theta),
        mode="lines",
        line=dict(color="rgba(255,0,0,0.3)", width=2, dash="dash"),
        name=f"Fitted circle (r={avg_r:.3f})",
        hoverinfo="skip",
    ))

    # Draw lines connecting consecutive tokens to show the winding
    for i in range(P):
        j = (i + 1) % P
        fig.add_trace(go.Scatter(
            x=[x_coords[i], x_coords[j]],
            y=[y_coords[i], y_coords[j]],
            mode="lines",
            line=dict(color="rgba(150,150,150,0.15)", width=0.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    circularity = pair_info["circularity_score"]
    freq_str = f" (freq k={pair_info['frequency']})" if pair_info.get("frequency") else ""

    fig.update_layout(
        title=f"{pair_info['description']}<br>Circularity: {circularity:.4f}{freq_str}",
        xaxis_title=pair_info["dim_x"] if isinstance(pair_info["dim_x"], str) else f"Dim {pair_info['dim_x']}",
        yaxis_title=pair_info["dim_y"] if isinstance(pair_info["dim_y"], str) else f"Dim {pair_info['dim_y']}",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        height=600,
        width=650,
    )

    return fig


def make_all_circles_summary(model: ModularAdditionTransformer,
                             key_frequencies: list[int] = None) -> go.Figure:
    """
    Create a summary figure showing ALL discovered circles in a grid.
    This is the "one-click" view that replaces manual dimension hunting.
    """
    pairs = find_circular_dimension_pairs(model, key_frequencies, top_k_pairs=3)

    n_plots = min(len(pairs), 8)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[p["description"][:50] for p in pairs[:n_plots]],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    P = model.P

    for idx, pair in enumerate(pairs[:n_plots]):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        x_coords = pair["x_coords"]
        y_coords = pair["y_coords"]

        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode="markers",
            marker=dict(size=4, color=np.arange(P), colorscale="hsv", showscale=False),
            hovertemplate="Token %{text}<extra></extra>",
            text=[str(i) for i in range(P)],
            showlegend=False,
        ), row=row, col=col)

        # Fitted circle
        cx, cy = x_coords.mean(), y_coords.mean()
        radii = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        avg_r = radii.mean()
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=cx + avg_r * np.cos(theta),
            y=cy + avg_r * np.sin(theta),
            mode="lines",
            line=dict(color="rgba(255,0,0,0.3)", width=1, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ), row=row, col=col)

    fig.update_layout(
        height=300 * n_rows,
        title_text="Auto-Discovered Circular Structures in Embedding Space",
    )

    return fig



# =============================================================================
# Training with Live Progress
# =============================================================================

def train_model(P: int = 113, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512,
                train_frac: float = 0.3, epochs: int = 80000, lr: float = 1e-3,
                weight_decay: float = 1.0, progress_cb=None, progress=None) -> tuple:
    """Train a model on modular addition until it groks."""
    model = ModularAdditionTransformer(P, d_model, n_heads, d_mlp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create dataset: all pairs (a, b) with target (a+b) mod P
    all_a = torch.arange(P).repeat_interleave(P)
    all_b = torch.arange(P).repeat(P)
    all_targets = (all_a + all_b) % P

    # Train/test split
    n_total = P * P
    n_train = int(n_total * train_frac)
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_a, train_b, train_t = all_a[train_idx], all_b[train_idx], all_targets[train_idx]
    test_a, test_b, test_t = all_a[test_idx], all_b[test_idx], all_targets[test_idx]

    best_test_acc = 0.0
    train_losses = []
    test_accs = []
    metrics_table = []

    def update(msg):
        if progress_cb:
            progress_cb(msg)

    for epoch in range(epochs):
        model.train()
        logits = model(train_a, train_b)
        loss = F.cross_entropy(logits, train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_a, test_b)
                test_preds = test_logits.argmax(dim=-1)
                test_acc = (test_preds == test_t).float().mean().item()
                train_preds = logits.argmax(dim=-1)
                train_acc = (train_preds == train_t).float().mean().item()

            train_losses.append((epoch, loss.item()))
            test_accs.append((epoch, test_acc))
            best_test_acc = max(best_test_acc, test_acc)

            metrics_table.append({
                "epoch": epoch,
                "train_loss": round(loss.item(), 6),
                "train_acc": round(train_acc, 4),
                "test_acc": round(test_acc, 4),
                "best_test_acc": round(best_test_acc, 4),
            })

            if progress is not None:
                pct = (epoch + 1) / epochs
                progress(pct, desc=f"Epoch {epoch}/{epochs} | loss={loss.item():.4f} | test_acc={test_acc:.3f}")

            update(f"Epoch {epoch}: loss={loss.item():.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

            if test_acc > 0.99:
                update(f"Grokked at epoch {epoch}!")
                if progress is not None:
                    progress(1.0, desc=f"Grokked at epoch {epoch}!")
                break

    # Save metrics table to CSV
    df = pd.DataFrame(metrics_table)
    csv_path = os.path.join(SAVE_DIR, "training_metrics.csv")
    df.to_csv(csv_path, index=False)

    model.eval()
    return model, train_losses, test_accs, metrics_table




# =============================================================================
# ACDC-Style Automatic Circuit Discovery
# =============================================================================

from computational_graph import ComputationalGraph

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

from acc_circuit_discoverer import ACDCCircuitDiscoverer

# =============================================================================
# Fourier Circuit Discovery (Nanda et al. 2023)
# =============================================================================

from discovered_circuit import DiscoveredCircuit
from circuit_discoverer import CircuitDiscoverer

# =============================================================================
# Combined Circuit Discovery: Fourier + ACDC
# =============================================================================

class CombinedCircuitDiscoverer:
    """
    Combines Fourier analysis (Nanda et al. 2023) with ACDC (Conmy et al. 2023)
    for comprehensive circuit discovery in grokking models.

    The Fourier analysis identifies WHAT the circuit computes (key frequencies,
    trig identities), while ACDC identifies HOW the circuit is structured
    (which components and connections are essential).
    """

    def __init__(self, model: ModularAdditionTransformer):
        self.model = model
        self.fourier_discoverer = CircuitDiscoverer(model)
        self.acdc_discoverer = ACDCCircuitDiscoverer(model, granularity="component")

    def full_discovery(self, acdc_threshold: float = 0.01, n_samples: int = 512,
                       progress_cb=None) -> dict:
        """
        Run both Fourier analysis and ACDC circuit discovery.

        Returns a comprehensive report combining both analyses.
        """
        def update(msg):
            if progress_cb:
                progress_cb(msg)

        # Phase 1: Fourier Analysis
        update("=" * 50)
        update("PHASE 1: Fourier Analysis (Nanda et al. 2023)")
        update("=" * 50)
        fourier_circuit = self.fourier_discoverer.full_discovery(progress_cb=progress_cb)

        # Phase 2: ACDC Circuit Discovery
        update("")
        update("=" * 50)
        update("PHASE 2: ACDC Circuit Discovery (Conmy et al. 2023)")
        update("=" * 50)
        acdc_circuit = self.acdc_discoverer.run_acdc(
            threshold=acdc_threshold,
            n_samples=n_samples,
            progress_cb=progress_cb,
        )

        # Phase 3: Cross-validation
        update("")
        update("=" * 50)
        update("PHASE 3: Cross-validation")
        update("=" * 50)

        acdc_summary = self.acdc_discoverer.get_circuit_summary(acdc_circuit)
        acdc_viz = self.acdc_discoverer.visualize_circuit(acdc_circuit)
        update(acdc_viz)

        # Check consistency: do ACDC-identified components align with Fourier analysis?
        consistency = self._check_consistency(fourier_circuit, acdc_summary)
        update(f"\nConsistency check:")
        update(f"  MLP pathway found by ACDC: {acdc_summary['has_mlp_path']}")
        update(f"  (Expected: True, since MLP computes trig identities)")
        update(f"  Attention heads in circuit: {acdc_summary['active_heads']}")
        update(f"  Fourier key frequencies: {fourier_circuit.key_frequencies}")

        return {
            "fourier_circuit": fourier_circuit,
            "acdc_circuit": acdc_circuit,
            "acdc_summary": acdc_summary,
            "acdc_visualization": acdc_viz,
            "consistency": consistency,
        }

    def _check_consistency(self, fourier_circuit: DiscoveredCircuit,
                           acdc_summary: dict) -> dict:
        """
        Check whether ACDC findings are consistent with Fourier analysis.
        """
        checks = {}

        # Check 1: MLP should be in the circuit (it computes trig identities)
        checks["mlp_in_circuit"] = acdc_summary["has_mlp_path"]

        # Check 2: At least some attention heads should be present
        # (they move information from input positions to output position)
        checks["attention_present"] = len(acdc_summary["active_heads"]) > 0

        # Check 3: Embeddings should connect to attention
        # (embeddings encode Fourier components that attention must access)
        checks["embed_to_attn"] = acdc_summary["edge_types"].get("embed_to_attn", 0) > 0

        # Check 4: High Fourier FVE suggests clean circuit
        checks["high_fourier_fve"] = fourier_circuit.fve_logits > 0.8

        # Check 5: Verification accuracy
        checks["high_verification_acc"] = fourier_circuit.verification_accuracy > 0.95

        checks["all_consistent"] = all(checks.values())

        return checks


# =============================================================================
# Ablation Tests
# =============================================================================

def ablation_test(model: ModularAdditionTransformer, key_freqs: list[int]) -> dict:
    """
    Per Section 4.4: ablate key frequencies and non-key frequencies
    to confirm the circuit is both necessary and sufficient.
    """
    P = model.P
    all_a = torch.arange(P).repeat_interleave(P)
    all_b = torch.arange(P).repeat(P)
    all_targets = (all_a + all_b) % P

    with torch.no_grad():
        logits = model(all_a, all_b)
        preds = logits.argmax(dim=-1)
        baseline_acc = (preds == all_targets).float().mean().item()

    results = {"baseline_accuracy": baseline_acc}

    with torch.no_grad():
        logits_np = model(all_a, all_b).cpu().numpy()

    a_vals = all_a.numpy()
    b_vals = all_b.numpy()

    logit_cube = logits_np.reshape(P, P, P)

    # DFT over first two axes
    logit_fft = np.fft.fft2(logit_cube, axes=(0, 1))

    # Ablate key frequencies (set them to zero) - should hurt
    logit_fft_ablated_key = logit_fft.copy()
    for k in key_freqs:
        logit_fft_ablated_key[k, :, :] = 0
        logit_fft_ablated_key[:, k, :] = 0
        logit_fft_ablated_key[P - k, :, :] = 0
        logit_fft_ablated_key[:, P - k, :] = 0

    logit_ablated_key = np.fft.ifft2(logit_fft_ablated_key, axes=(0, 1)).real
    preds_ablated_key = logit_ablated_key.reshape(P * P, P).argmax(axis=1)
    targets_np = all_targets.numpy()
    acc_without_key = (preds_ablated_key == targets_np).mean()
    results["accuracy_without_key_freqs"] = float(acc_without_key)

    # Ablate everything EXCEPT key frequencies (should preserve performance)
    logit_fft_restricted = np.zeros_like(logit_fft)
    logit_fft_restricted[0, 0, :] = logit_fft[0, 0, :]
    for k in key_freqs:
        logit_fft_restricted[k, :, :] = logit_fft[k, :, :]
        logit_fft_restricted[:, k, :] = logit_fft[:, k, :]
        logit_fft_restricted[P - k, :, :] = logit_fft[P - k, :, :]
        logit_fft_restricted[:, P - k, :] = logit_fft[:, P - k, :]

    logit_restricted = np.fft.ifft2(logit_fft_restricted, axes=(0, 1)).real
    preds_restricted = logit_restricted.reshape(P * P, P).argmax(axis=1)
    acc_restricted = (preds_restricted == targets_np).mean()
    results["accuracy_restricted_to_key_freqs"] = float(acc_restricted)

    # Per-frequency ablation
    per_freq_results = {}
    for k in key_freqs:
        logit_fft_single = logit_fft.copy()
        logit_fft_single[k, :, :] = 0
        logit_fft_single[:, k, :] = 0
        logit_fft_single[P - k, :, :] = 0
        logit_fft_single[:, P - k, :] = 0
        logit_single = np.fft.ifft2(logit_fft_single, axes=(0, 1)).real
        preds_single = logit_single.reshape(P * P, P).argmax(axis=1)
        acc_single = (preds_single == targets_np).mean()
        per_freq_results[k] = float(acc_single)

    results["per_frequency_ablation"] = per_freq_results
    return results


# =============================================================================
# Visualization Helpers (Plotly)
# =============================================================================

def make_training_plot(train_losses: list, test_accs: list) -> go.Figure:
    """Create a live training progress plot with loss and test accuracy."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Training Loss", "Test Accuracy"),
        vertical_spacing=0.12,
    )

    if train_losses:
        epochs_loss, losses = zip(*train_losses)
        fig.add_trace(
            go.Scatter(x=list(epochs_loss), y=list(losses), mode="lines",
                       name="Train Loss", line=dict(color="red", width=2)),
            row=1, col=1,
        )

    if test_accs:
        epochs_acc, accs = zip(*test_accs)
        fig.add_trace(
            go.Scatter(x=list(epochs_acc), y=list(accs), mode="lines",
                       name="Test Accuracy", line=dict(color="blue", width=2)),
            row=2, col=1,
        )
        # Add grokking threshold line
        fig.add_hline(y=0.99, line_dash="dash", line_color="green",
                      annotation_text="Grokking threshold (99%)", row=2, col=1)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=2, col=1)
    fig.update_layout(height=600, title_text="Training Progress", showlegend=True)

    return fig


def make_fourier_plot(embed_norms: np.ndarray, wl_norms: np.ndarray, key_freqs: list[int]) -> go.Figure:
    """Plot Fourier norms of embedding and neuron-logit map."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Embedding Fourier Norms (W_E)", "Neuron-Logit Map Fourier Norms (W_L)"),
        vertical_spacing=0.12,
    )

    freqs = list(range(len(embed_norms)))

    # Highlight key frequencies
    colors_embed = ["red" if f in key_freqs else "steelblue" for f in freqs]
    colors_wl = ["red" if f in key_freqs else "steelblue" for f in freqs]

    fig.add_trace(
        go.Bar(x=freqs, y=embed_norms, marker_color=colors_embed, name="W_E norms"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=freqs, y=wl_norms, marker_color=colors_wl, name="W_L norms"),
        row=2, col=1,
    )

    fig.update_xaxes(title_text="Frequency k", row=1, col=1)
    fig.update_xaxes(title_text="Frequency k", row=2, col=1)
    fig.update_yaxes(title_text="Norm", row=1, col=1)
    fig.update_yaxes(title_text="Norm", row=2, col=1)
    fig.update_layout(height=600, title_text="Fourier Analysis (red = key frequencies)", showlegend=False)

    return fig


def make_neuron_assignment_plot(assignments: dict, key_freqs: list[int], d_mlp: int) -> go.Figure:
    """Visualize neuron-to-frequency assignments."""
    freq_counts = {k: 0 for k in key_freqs}
    freq_counts["unassigned"] = 0

    for neuron_idx, info in assignments.items():
        freq = info["frequency"]
        if freq in freq_counts:
            freq_counts[freq] += 1

    freq_counts["unassigned"] = d_mlp - len(assignments)

    labels = [str(k) for k in key_freqs] + ["unassigned"]
    values = [freq_counts.get(k, 0) for k in key_freqs] + [freq_counts["unassigned"]]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title_text=f"Neuron Frequency Assignments ({len(assignments)}/{d_mlp} assigned)")

    return fig


def make_ablation_plot(ablation_results: dict) -> go.Figure:
    """Visualize ablation test results."""
    labels = ["Baseline", "Without Key Freqs", "Only Key Freqs"]
    values = [
        ablation_results["baseline_accuracy"],
        ablation_results["accuracy_without_key_freqs"],
        ablation_results["accuracy_restricted_to_key_freqs"],
    ]

    colors = ["green", "red", "blue"]

    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=colors)])
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05])
    fig.update_layout(title_text="Ablation Test: Key Frequencies are Necessary & Sufficient", height=400)

    return fig
def make_embedding_circle_plot(model: ModularAdditionTransformer, dim_x: int = 0, dim_y: int = 1, show_all_tokens: bool = True) -> go.Figure:
    """
    Plot token embeddings projected onto a chosen pair of dimensions.
    Shows circles with real coordinate values for each token.
    
    For a grokked model, pairs of dimensions corresponding to key frequencies
    will show tokens arranged in circles (Fourier structure).
    """
    P = model.P
    W_E = model.embed.weight[:P].detach().cpu().numpy()  # (P, d_model)
    
    x_coords = W_E[:, dim_x]
    y_coords = W_E[:, dim_y]
    
    fig = go.Figure()
    
    # Plot all token positions
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers+text",
        marker=dict(size=10, color=np.arange(P), colorscale="hsv", showscale=True, 
                    colorbar=dict(title="Token ID")),
        text=[str(i) for i in range(P)],
        textposition="top center",
        textfont=dict(size=7),
        customdata=np.stack([
            np.arange(P),
            x_coords,
            y_coords,
        ], axis=-1),
        hovertemplate=(
            "Token: %{customdata[0]:.0f}<br>"
            f"Dim {dim_x}: %{{customdata[1]:.4f}}<br>"
            f"Dim {dim_y}: %{{customdata[2]:.4f}}<br>"
            "<extra></extra>"
        ),
        name="Token Embeddings",
    ))
    
    # Fit and draw a circle to show the circular structure
    center_x, center_y = x_coords.mean(), y_coords.mean()
    radii = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    avg_radius = radii.mean()
    
    theta = np.linspace(0, 2*np.pi, 200)
    circle_x = center_x + avg_radius * np.cos(theta)
    circle_y = center_y + avg_radius * np.sin(theta)
    
    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode="lines",
        line=dict(color="rgba(255,0,0,0.3)", width=2, dash="dash"),
        name=f"Fitted circle (r={avg_radius:.3f})",
        hoverinfo="skip",
    ))
    
    # Add annotations showing the actual values for a few tokens
    for token_id in [0, P//4, P//2, 3*P//4]:
        fig.add_annotation(
            x=x_coords[token_id], y=y_coords[token_id],
            text=f"t={token_id}<br>({x_coords[token_id]:.3f}, {y_coords[token_id]:.3f})",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1,
            font=dict(size=9, color="red"),
        )
    
    fig.update_layout(
        title=f"Embedding Space: Dim {dim_x} vs Dim {dim_y} (P={P})",
        xaxis_title=f"Dimension {dim_x}",
        yaxis_title=f"Dimension {dim_y}",
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal aspect ratio
        height=650,
        width=700,
    )
    
    return fig


def make_embedding_all_dims_heatmap(model: ModularAdditionTransformer, token_id: int = 0) -> go.Figure:
    """
    Show ALL embedding dimensions for a selected token as a heatmap/bar chart.
    This lets you inspect every dimension's value for any token.
    """
    P = model.P
    W_E = model.embed.weight[:P].detach().cpu().numpy()  # (P, d_model)
    d_model = W_E.shape[1]
    
    token_embedding = W_E[token_id]  # (d_model,)
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=(
                            f"All {d_model} Embedding Dimensions for Token {token_id}",
                            f"Embedding Heatmap (Tokens × Dimensions)"
                        ),
                        row_heights=[0.4, 0.6],
                        vertical_spacing=0.12)
    
    # Bar chart of all dimensions for selected token
    colors = ["red" if abs(v) > np.std(token_embedding) * 2 else "steelblue" 
              for v in token_embedding]
    
    fig.add_trace(go.Bar(
        x=list(range(d_model)),
        y=token_embedding,
        marker_color=colors,
        name=f"Token {token_id}",
        hovertemplate="Dim %{x}: %{y:.4f}<extra></extra>",
    ), row=1, col=1)
    
    # Heatmap of all tokens × all dimensions (or a subset)
    # Show a window of tokens around the selected one
    window = min(30, P)
    start = max(0, token_id - window//2)
    end = min(P, start + window)
    
    fig.add_trace(go.Heatmap(
        z=W_E[start:end, :],
        x=list(range(d_model)),
        y=[str(i) for i in range(start, end)],
        colorscale="RdBu_r",
        zmid=0,
        hovertemplate="Token %{y}, Dim %{x}: %{z:.4f}<extra></extra>",
        name="Embedding Matrix",
    ), row=2, col=1)
    
    fig.update_xaxes(title_text="Dimension", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_xaxes(title_text="Dimension", row=2, col=1)
    fig.update_yaxes(title_text="Token ID", row=2, col=1)
    fig.update_layout(height=800, showlegend=False)
    
    return fig

def make_layer_activation_plots(model: ModularAdditionTransformer, a: int, b: int) -> dict:
    """
    Run a single (a, b) input through the model and capture what comes out
    of EVERY layer and sub-component. Returns a dict of plotly figures.

    Shows:
    - Token + positional embeddings
    - Each attention head's output (per-head)
    - Attention weights (which positions attend to which)
    - Residual stream after attention
    - MLP pre-activations (before ReLU)
    - MLP hidden activations (after ReLU)
    - MLP output
    - Final residual stream
    - Logits
    """
    P = model.P
    a_tensor = torch.tensor([a])
    b_tensor = torch.tensor([b])

    with torch.no_grad():
        logits, activations = model.forward_with_hooks(a_tensor, b_tensor)

    figures = {}
    correct = (a + b) % P

    # --- 1. Embeddings ---
    tok_emb = activations["tok_embed"][0].numpy()  # (3, d_model)
    pos_emb = activations["pos_embed"][0].numpy()  # (3, d_model)
    combined_emb = activations["embed"][0].numpy()  # (3, d_model)

    fig_embed = make_subplots(rows=3, cols=1,
                              subplot_titles=("Token Embedding", "Positional Embedding", "Combined (Token + Pos)"),
                              vertical_spacing=0.08)

    for row, (data, name) in enumerate([(tok_emb, "Token"), (pos_emb, "Position"), (combined_emb, "Combined")], 1):
        for pos_idx, pos_name in enumerate(["a", "b", "="]):
            fig_embed.add_trace(go.Scatter(
                y=data[pos_idx], mode="lines", name=f"{pos_name} ({name})",
                hovertemplate=f"Pos={pos_name} Dim %{{x}}: %{{y:.4f}}<extra></extra>",
            ), row=row, col=1)

    fig_embed.update_layout(height=700, title=f"Embeddings for a={a}, b={b}")
    figures["embeddings"] = fig_embed

    # --- 2. Attention Weights ---
    attn_weights = activations["attn_weights"][0].numpy()  # (n_heads, 1, 2)

    fig_attn = go.Figure()
    head_names = [f"Head {h}" for h in range(model.n_heads)]
    attn_to_a = attn_weights[:, 0, 0]  # attention to position 0 (token a)
    attn_to_b = attn_weights[:, 0, 1]  # attention to position 1 (token b)

    fig_attn.add_trace(go.Bar(name="Attn to 'a'", x=head_names, y=attn_to_a, marker_color="blue"))
    fig_attn.add_trace(go.Bar(name="Attn to 'b'", x=head_names, y=attn_to_b, marker_color="orange"))
    fig_attn.update_layout(barmode="group", title=f"Attention Weights (from '=' to a,b) | a={a}, b={b}",
                           yaxis_title="Weight", height=350)
    figures["attention_weights"] = fig_attn

    # --- 3. Per-Head Attention Outputs ---
    fig_heads = make_subplots(rows=model.n_heads, cols=1,
                              subplot_titles=[f"Head {h} Output" for h in range(model.n_heads)],
                              vertical_spacing=0.05)

    for h in range(model.n_heads):
        head_out = activations[f"attn_head_{h}"][0, 0].numpy()  # (d_head,)
        fig_heads.add_trace(go.Bar(
            y=head_out, name=f"Head {h}",
            hovertemplate=f"Head {h} Dim %{{x}}: %{{y:.4f}}<extra></extra>",
        ), row=h+1, col=1)

    fig_heads.update_layout(height=200*model.n_heads, title="Per-Head Attention Outputs", showlegend=False)
    figures["attn_head_outputs"] = fig_heads

    # --- 4. Combined Attention Output ---
    attn_out = activations["attn_out"][0, 0].numpy()  # (d_model,)
    fig_attn_out = go.Figure(go.Bar(y=attn_out,
                                     hovertemplate="Dim %{x}: %{y:.4f}<extra></extra>"))
    fig_attn_out.update_layout(title=f"Combined Attention Output (W_O applied) | a={a}, b={b}",
                               height=300, xaxis_title="Dimension", yaxis_title="Value")
    figures["attn_combined_output"] = fig_attn_out

    # --- 5. Residual Stream (mid) ---
    residual_mid = activations["residual_mid"][0, 0].numpy()  # (d_model,)
    fig_res_mid = go.Figure(go.Bar(y=residual_mid,
                                    hovertemplate="Dim %{x}: %{y:.4f}<extra></extra>"))
    fig_res_mid.update_layout(title=f"Residual Stream (after attention, before MLP) | a={a}, b={b}",
                              height=300, xaxis_title="Dimension", yaxis_title="Value")
    figures["residual_mid"] = fig_res_mid

    # --- 6. MLP Pre-activations (before ReLU) ---
    mlp_pre = activations["mlp_pre"][0, 0].numpy()  # (d_mlp,)
    fig_mlp_pre = go.Figure()
    colors_pre = ["green" if v > 0 else "red" for v in mlp_pre]
    fig_mlp_pre.add_trace(go.Bar(y=mlp_pre, marker_color=colors_pre,
                                  hovertemplate="Neuron %{x}: %{y:.4f}<extra></extra>"))
    fig_mlp_pre.update_layout(title=f"MLP Pre-Activations (before ReLU) | a={a}, b={b} | "
                              f"{(mlp_pre > 0).sum()}/{len(mlp_pre)} positive",
                              height=350, xaxis_title="Neuron Index", yaxis_title="Pre-activation")
    figures["mlp_pre"] = fig_mlp_pre

    # --- 7. MLP Hidden (after ReLU) ---
    mlp_hidden = activations["mlp_hidden"][0, 0].numpy()  # (d_mlp,)
    fig_mlp_hidden = go.Figure()
    # Highlight active neurons
    active_mask = mlp_hidden > 0
    fig_mlp_hidden.add_trace(go.Bar(
        y=mlp_hidden,
        marker_color=["green" if a else "lightgray" for a in active_mask],
        hovertemplate="Neuron %{x}: %{y:.4f}<extra></extra>",
    ))
    fig_mlp_hidden.update_layout(
        title=f"MLP Hidden (after ReLU) | a={a}, b={b} | "
              f"{active_mask.sum()}/{len(mlp_hidden)} active neurons",
        height=350, xaxis_title="Neuron Index", yaxis_title="Activation",
    )
    figures["mlp_hidden"] = fig_mlp_hidden

    # --- 8. MLP Output ---
    mlp_out = activations["mlp_out"][0, 0].numpy()  # (d_model,)
    fig_mlp_out = go.Figure(go.Bar(y=mlp_out,
                                    hovertemplate="Dim %{x}: %{y:.4f}<extra></extra>"))
    fig_mlp_out.update_layout(title=f"MLP Output (projected back to d_model) | a={a}, b={b}",
                              height=300, xaxis_title="Dimension", yaxis_title="Value")
    figures["mlp_output"] = fig_mlp_out

    # --- 9. Final Residual Stream ---
    residual_final = activations["residual_final"][0, 0].numpy()  # (d_model,)
    fig_res_final = go.Figure(go.Bar(y=residual_final,
                                      hovertemplate="Dim %{x}: %{y:.4f}<extra></extra>"))
    fig_res_final.update_layout(title=f"Final Residual Stream (before unembed) | a={a}, b={b}",
                                height=300, xaxis_title="Dimension", yaxis_title="Value")
    figures["residual_final"] = fig_res_final

    # --- 10. Logits ---
    logits_np = activations["logits"][0].numpy()  # (P,)
    top_k = 10
    top_indices = np.argsort(logits_np)[-top_k:][::-1]

    fig_logits = go.Figure()
    colors_logits = ["green" if idx == correct else "steelblue" for idx in range(P)]
    fig_logits.add_trace(go.Bar(
        x=list(range(P)), y=logits_np, marker_color=colors_logits,
        hovertemplate="Class %{x}: %{y:.4f}<extra></extra>",
    ))
    fig_logits.add_annotation(x=correct, y=logits_np[correct],
                              text=f"CORRECT: {correct}", showarrow=True, arrowhead=2)
    fig_logits.update_layout(
        title=f"Output Logits | a={a}, b={b} | correct={(a+b)%P} | predicted={logits_np.argmax()}",
        height=350, xaxis_title="Output Class", yaxis_title="Logit Value",
    )
    figures["logits"] = fig_logits

    return figures

def make_acdc_circuit_plot(acdc_summary: dict) -> go.Figure:
    """Visualize the ACDC-discovered circuit as a network diagram using Plotly."""
    # Create a simple layered graph visualization
    active_nodes = acdc_summary["active_nodes"]
    edge_details = acdc_summary["edge_details"]

    # Assign positions based on layer
    layer_positions = {
        "tok_embed": (0, 0),
        "pos_embed": (0, 1),
    }

    # Attention heads in layer 1
    attn_heads = [n for n in active_nodes if n.startswith("attn_head_")]
    for i, h in enumerate(sorted(attn_heads)):
        layer_positions[h] = (1, i * 0.8)

    # MLP in layer 2
    if "mlp" in active_nodes:
        layer_positions["mlp"] = (2, 0.5)

    # Neuron groups
    neuron_groups = [n for n in active_nodes if n.startswith("neuron_group_")]
    for i, ng in enumerate(sorted(neuron_groups)):
        layer_positions[ng] = (2, i * 0.3)

    # Output in layer 3
    if "unembed" in active_nodes:
        layer_positions["unembed"] = (3, 0.5)

    # Other nodes
    for n in active_nodes:
        if n not in layer_positions:
            layer_positions[n] = (1.5, len(layer_positions) * 0.3)

    fig = go.Figure()

    # Draw edges
    all_edges = []
    for edge_type, edges in edge_details.items():
        all_edges.extend(edges)

    edge_x = []
    edge_y = []
    for parent, child in all_edges:
        if parent in layer_positions and child in layer_positions:
            x0, y0 = layer_positions[parent]
            x1, y1 = layer_positions[child]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="rgba(100,100,100,0.5)"),
        hoverinfo="none",
        name="Edges",
    ))

    # Draw nodes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []

    color_map = {
        "tok_embed": "orange",
        "pos_embed": "orange",
        "mlp": "green",
        "unembed": "purple",
    }

    for node in active_nodes:
        if node in layer_positions:
            x, y = layer_positions[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            if node in color_map:
                node_colors.append(color_map[node])
            elif node.startswith("attn_head_"):
                node_colors.append("blue")
            elif node.startswith("neuron_group_"):
                node_colors.append("lightgreen")
            else:
                node_colors.append("gray")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=20, color=node_colors, line=dict(width=2, color="black")),
        text=node_text,
        textposition="top center",
        textfont=dict(size=9),
        name="Nodes",
    ))

    fig.update_layout(
        title=f"ACDC Circuit ({acdc_summary['num_edges']} edges)",
        xaxis=dict(title="Layer", showgrid=False),
        yaxis=dict(title="", showgrid=False),
        height=500,
        showlegend=False,
    )

    return fig


# =============================================================================
# Gradio GUI
# =============================================================================

def build_gui():
    """Build the Gradio interface with live training plots and metrics table."""
    import gradio as gr

    # Global state
    state = {
        "model": None,
        "circuit": None,
        "train_losses": [],
        "test_accs": [],
        "metrics_table": [],
        "ablation_results": None,
        "acdc_result": None,
    }

    def load_saved_metrics():
        """Load previously saved training metrics if available."""
        csv_path = os.path.join(SAVE_DIR, "training_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        return pd.DataFrame()

    def train_and_update(P, d_model, n_heads, d_mlp, train_frac, epochs, lr, weight_decay, progress=gr.Progress()):
        """Train model with live plot updates using a generator."""
        P, d_model, n_heads, d_mlp, epochs = int(P), int(d_model), int(n_heads), int(d_mlp), int(epochs)
        train_frac, lr, weight_decay = float(train_frac), float(lr), float(weight_decay)

        logs = []
        train_losses = []
        test_accs = []
        metrics_table = []

        model = ModularAdditionTransformer(P=P, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Create dataset: all pairs (a, b) with target (a+b) mod P
        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        all_targets = (all_a + all_b) % P

        # Train/test split
        n_total = P * P
        n_train = int(n_total * train_frac)
        perm = torch.randperm(n_total)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        train_a, train_b, train_t = all_a[train_idx], all_b[train_idx], all_targets[train_idx]
        test_a, test_b, test_t = all_a[test_idx], all_b[test_idx], all_targets[test_idx]

        best_test_acc = 0.0

        for epoch in range(epochs):
            model.train()
            logits = model(train_a, train_b)
            loss = F.cross_entropy(logits, train_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    test_logits = model(test_a, test_b)
                    test_preds = test_logits.argmax(dim=-1)
                    test_acc = (test_preds == test_t).float().mean().item()
                    train_preds = logits.argmax(dim=-1)
                    train_acc = (train_preds == train_t).float().mean().item()

                train_losses.append((epoch, loss.item()))
                test_accs.append((epoch, test_acc))
                best_test_acc = max(best_test_acc, test_acc)

                metrics_table.append({
                    "epoch": epoch,
                    "train_loss": round(loss.item(), 6),
                    "train_acc": round(train_acc, 4),
                    "test_acc": round(test_acc, 4),
                    "best_test_acc": round(best_test_acc, 4),
                })

                log_msg = f"Epoch {epoch}: loss={loss.item():.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}"
                logs.append(log_msg)

                progress((epoch + 1) / epochs, desc=log_msg)

                # Yield live updates
                fig = make_training_plot(train_losses, test_accs)
                df = pd.DataFrame(metrics_table)
                yield fig, df, "\n".join(logs[-20:])

                if test_acc > 0.99:
                    logs.append(f"Grokked at epoch {epoch}!")
                    break

        # Save and update state
        model.eval()
        state["model"] = model
        state["train_losses"] = train_losses
        state["test_accs"] = test_accs
        state["metrics_table"] = metrics_table

        config = {"P": P, "d_model": d_model, "n_heads": n_heads, "d_mlp": d_mlp,
                  "train_frac": train_frac, "epochs": epochs, "lr": lr, "weight_decay": weight_decay}
        save_run(model, train_losses, test_accs, metrics_table, config)

        fig = make_training_plot(train_losses, test_accs)
        df = pd.DataFrame(metrics_table)
        yield fig, df, "\n".join(logs[-20:])

    def generate_latex_equations(progress=gr.Progress()):
        """Generate full LaTeX equations for the discovered circuit."""
        if state["model"] is None or state["circuit"] is None:
            return "No circuit discovered yet! Run Fourier Discovery first."

        circuit = state["circuit"]
        generator = CircuitLatexGenerator(
            P=state["model"].P,
            key_frequencies=circuit.key_frequencies,
            neuron_assignments=circuit.neuron_frequency_assignments,
        )

        full_latex = generator.full_circuit_latex()

        # Also generate per-frequency detailed equations
        per_freq_latex = []
        for k in circuit.key_frequencies:
            per_freq_latex.append(generator.per_frequency_detail(k))

        full_output = full_latex + "\n\n" + "\n\n".join(per_freq_latex)

        # Also generate a "quick reference" card with the key equations
        quick_ref = generate_quick_reference(
            P=state["model"].P,
            key_frequencies=circuit.key_frequencies,
            fve=circuit.fve_logits,
            verification_acc=circuit.verification_accuracy,
            neuron_assignments=circuit.neuron_frequency_assignments,
        )

        return full_output, quick_ref


    def generate_quick_reference(P: int, key_frequencies: list, fve: float,
                                  verification_acc: float, neuron_assignments: dict) -> str:
        """Generate a concise quick-reference card summarizing the circuit equations."""
        freqs_str = ", ".join(str(k) for k in key_frequencies)
        n_assigned = len(neuron_assignments)

        # Count neurons per frequency
        freq_neuron_counts = {}
        for info in neuron_assignments.values():
            freq = info.get("frequency")
            freq_neuron_counts[freq] = freq_neuron_counts.get(freq, 0) + 1

        neuron_breakdown = ", ".join(
            f"k={k}: {freq_neuron_counts.get(k, 0)} neurons"
            for k in key_frequencies
        )

        quick_ref = (
            r"\section*{Quick Reference Card}" + "\n\n"
            r"\begin{tcolorbox}[colback=blue!5!white, colframe=blue!75!black, title=Circuit Summary]" + "\n"
            f"  \\textbf{{Task:}} $(a + b) \\bmod {P}$ \\hfill "
            f"\\textbf{{FVE:}} {fve:.4f} \\hfill "
            f"\\textbf{{Accuracy:}} {verification_acc*100:.2f}\\%\n\n"
            f"  \\textbf{{Key frequencies:}} $\\mathcal{{K}} = \\{{{freqs_str}\\}}$ "
            f"\\quad ({len(key_frequencies)} frequencies)\n\n"
            f"  \\textbf{{Neurons assigned:}} {n_assigned}/512 \\quad ({neuron_breakdown})\n"
            r"\end{tcolorbox}" + "\n\n"
            r"\begin{tcolorbox}[colback=green!5!white, colframe=green!75!black, title=The Core Equation]" + "\n"
            r"  \begin{equation*}" + "\n"
            r"    \boxed{" + "\n"
            r"      \text{Logit}(c \mid a, b) = "
            r"\sum_{k \in \mathcal{K}} "
            r"\underbrace{\alpha_k}_{\substack{\text{learned} \\ \text{amplitude}}} "
            r"\cdot \underbrace{\cos\!\left(\frac{2\pi k (a + b - c)}{" + str(P) + r"}\right)}_{"
            r"\substack{\text{max at } c = (a+b) \bmod " + str(P) + r" \\"
            r"\text{(constructive interference)}}}"
            r"    }" + "\n"
            r"  \end{equation*}" + "\n"
            r"\end{tcolorbox}" + "\n\n"
            r"\begin{tcolorbox}[colback=orange!5!white, colframe=orange!75!black, title=Data Flow]" + "\n"
            r"  \begin{equation*}" + "\n"
            r"    \underbrace{a, b}_{\text{inputs}}"
            r" \xrightarrow{\quad W_E \quad} "
            r"\underbrace{\cos(\omega_k t), \sin(\omega_k t)}_{\text{Fourier embedding}}"
            r" \xrightarrow{\quad \text{Attn} \quad} "
            r"\underbrace{\text{degree-2 products}}_{\substack{\text{attn weight} \times \\ \text{OV output}}}"
            r" \xrightarrow{\quad \text{MLP} \quad} "
            r"\underbrace{\cos(\omega_k(a+b)), \sin(\omega_k(a+b))}_{\text{trig addition identities}}"
            r" \xrightarrow{\quad W_U \quad} "
            r"\underbrace{\cos(\omega_k(a+b-c))}_{\text{logits}}" + "\n"
            r"  \end{equation*}" + "\n"
            r"\end{tcolorbox}"
        )

        return quick_ref

    def render_latex_to_display(progress=gr.Progress()):
        """
        Generate LaTeX equations and render them for display in the GUI.
        Returns both raw LaTeX (for copy-paste into a .tex file) and
        a Markdown-rendered version for in-app viewing.
        """
        if state["model"] is None or state["circuit"] is None:
            return (
                "⚠️ No circuit discovered yet! Run Fourier Discovery first.",
                "No LaTeX generated.",
                "No quick reference generated.",
            )

        circuit = state["circuit"]
        progress(0.2, desc="Generating LaTeX equations...")

        generator = CircuitLatexGenerator(
            P=state["model"].P,
            key_frequencies=circuit.key_frequencies,
            neuron_assignments=circuit.neuron_frequency_assignments,
        )

        progress(0.5, desc="Building full equation set...")
        full_latex = generator.full_circuit_latex()

        # Per-frequency details
        per_freq_sections = []
        for k in circuit.key_frequencies:
            per_freq_sections.append(generator.per_frequency_detail(k))

        full_latex_with_details = full_latex + "\n\n" + "\n\n".join(per_freq_sections)

        progress(0.7, desc="Generating quick reference...")
        quick_ref = generate_quick_reference(
            P=state["model"].P,
            key_frequencies=circuit.key_frequencies,
            fve=circuit.fve_logits,
            verification_acc=circuit.verification_accuracy,
            neuron_assignments=circuit.neuron_frequency_assignments,
        )

        progress(0.9, desc="Formatting for display...")

        P = state["model"].P
        freqs_str = ", ".join(str(k) for k in circuit.key_frequencies)

        display_md = f"""## Circuit Equations for $(a + b) \\bmod {P}$

### Key Frequencies

$$\\mathcal{{K}} = \\{{{freqs_str}\\}}$$

### Step 1: Embedding

$$\\mathbf{{x}}^{{(0)}}_a = \\underbrace{{W_E \\cdot \\mathbf{{e}}_a}}_{{\\text{{token embedding of }} a}} + \\underbrace{{\\mathbf{{p}}_0}}_{{\\text{{positional embedding (pos 0)}}}}$$

$$\\approx \\sum_{{k \\in \\mathcal{{K}}}} \\left[ \\underbrace{{\\alpha_k \\cos\\!\\left(\\frac{{2\\pi k \\cdot a}}{{{P}}}\\right)}}_{{\\text{{cosine from }} W_E}} \\cdot \\mathbf{{u}}_k^{{(\\cos)}} + \\underbrace{{\\beta_k \\sin\\!\\left(\\frac{{2\\pi k \\cdot a}}{{{P}}}\\right)}}_{{\\text{{sine from }} W_E}} \\cdot \\mathbf{{u}}_k^{{(\\sin)}} \\right]$$

### Step 2: Attention (Move Info to Output Position)

$$A^{{(j)}}_0 = \\underbrace{{\\sigma\\!\\left(\\text{{score}}^{{(j)}}_{{=\\to a}} - \\text{{score}}^{{(j)}}_{{=\\to b}}\\right)}}_{{\\text{{softmax over 2 elements = sigmoid of difference}}}}$$

$$\\approx \\underbrace{{0.5}}_{{\\text{{uniform}}}} + \\underbrace{{\\gamma_j \\left(\\cos(\\omega_{{k_j}} a) - \\cos(\\omega_{{k_j}} b)\\right)}}_{{\\text{{periodic modulation from }} W_E^\\top W_K^{{\\top}} W_Q \\mathbf{{x}}_=}}$$

### Step 3: MLP (Trig Identities)

$$\\underbrace{{\\mathbf{{u}}_k^\\top \\cdot \\text{{MLP}}(a,b)}}_{{\\text{{project onto direction in }} W_L}} \\approx \\overbrace{{\\underbrace{{\\cos(\\omega_k a)\\cos(\\omega_k b)}}_{{\\text{{from attn products}}}} - \\underbrace{{\\sin(\\omega_k a)\\sin(\\omega_k b)}}_{{\\text{{from attn products}}}}}}^{{= \\cos(\\omega_k(a+b)) \\text{{ (addition formula)}}}}$$

### Step 4: Unembedding (Fourier to Logits)

$$\\text{{Logit}}(c \\mid a, b) \\approx \\sum_{{k \\in \\mathcal{{K}}}} \\underbrace{{\\alpha_k}}_{{\\text{{amplitude}}}} \\underbrace{{\\cos\\!\\left(\\frac{{2\\pi k(a+b-c)}}{{{P}}}\\right)}}_{{\\substack{{\\text{{peaks when }} c \\equiv a+b \\pmod{{{P}}}}} \\\\ \\text{{since }} \\cos(0) = 1}}$$

### Step 5: Prediction

$$\\hat{{c}} = \\underbrace{{\\arg\\max_c}}_{{\\text{{select max logit}}}} \\sum_{{k \\in \\mathcal{{K}}}} \\alpha_k \\cos\\!\\left(\\frac{{2\\pi k(a+b-c)}}{{{P}}}\\right) = \\underbrace{{(a+b) \\bmod {P}}}_{{\\text{{constructive interference at correct answer}}}}$$

### Verification

| Metric | Value |
|:-------|:------|
| FVE (Fraction of Variance Explained) | {circuit.fve_logits:.4f} |
| Exhaustive verification accuracy | {circuit.verification_accuracy*100:.2f}% |
| Neurons assigned to key frequencies | {len(circuit.neuron_frequency_assignments)}/512 |

### Per-Frequency Neuron Counts

"""
        # Add per-frequency neuron counts using block-level math on separate lines
        freq_neuron_counts = {}
        for info in circuit.neuron_frequency_assignments.values():
            freq = info.get("frequency")
            freq_neuron_counts[freq] = freq_neuron_counts.get(freq, 0) + 1

        for k in circuit.key_frequencies:
            count = freq_neuron_counts.get(k, 0)
            display_md += f"**Frequency {k}:** {count} neurons\n\n"
            display_md += f"$$\\omega_{{{k}}} = \\frac{{2\\pi \\cdot {k}}}{{{P}}}$$\n\n"

        progress(1.0, desc="Done!")

        # Now build the full compilable LaTeX document for the raw output
        full_latex_document = _build_full_latex_document(P, circuit, full_latex_with_details)

        return display_md, full_latex_document, quick_ref


    def _build_full_latex_document(P: int, circuit, body_latex: str) -> str:
        """Build a complete, compilable LaTeX document with fourier font and detailed equations."""
        freqs_str = ", ".join(str(k) for k in circuit.key_frequencies)

        # Count neurons per frequency
        freq_neuron_counts = {}
        for info in circuit.neuron_frequency_assignments.values():
            freq = info.get("frequency")
            freq_neuron_counts[freq] = freq_neuron_counts.get(freq, 0) + 1

        preamble = r"""\documentclass[11pt,a4paper]{article}

% === Fonts ===
\usepackage{fourier}  % Utopia-based font with matching math (loads T1 fontenc internally)
\usepackage[utf8]{inputenc}

% === Math ===
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{mathtools}

% === Layout ===
\usepackage[margin=2cm]{geometry}
\usepackage{parskip}

% === Colors and boxes ===
\usepackage[dvipsnames]{xcolor}
\usepackage[most]{tcolorbox}

% === Hyperlinks ===
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}

% === Allow line breaks in equations ===
\allowdisplaybreaks

% === Custom commands ===
\newcommand{\R}{\mathbb{R}}
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bu}{\mathbf{u}}
\newcommand{\bv}{\mathbf{v}}
\newcommand{\bp}{\mathbf{p}}
\newcommand{\be}{\mathbf{e}}
\newcommand{\MLP}{\operatorname{MLP}}
\newcommand{\attn}{\operatorname{Attn}}
\newcommand{\Logit}{\operatorname{Logit}}
\newcommand{\FVE}{\operatorname{FVE}}
\newcommand{\ReLU}{\operatorname{ReLU}}
\DeclareMathOperator*{\argmax}{arg\,max}

\title{Fourier Multiplication Circuit\\[6pt]
\large Discovered in a 1-Layer Transformer Trained on $(a + b) \bmod """ + str(P) + r"""$}
\author{Auto-generated by Grokking Circuit Discovery Tool}
\date{\today}

\begin{document}
\maketitle

\begin{tcolorbox}[colback=blue!5!white, colframe=blue!75!black, title=Circuit Summary]
  \textbf{Task:} $(a + b) \bmod """ + str(P) + r"""$ \hfill
  \textbf{FVE:} """ + f"{circuit.fve_logits:.4f}" + r""" \hfill
  \textbf{Formula Accuracy:} """ + f"{circuit.verification_accuracy*100:.2f}" + r"""\%

  \textbf{Key frequencies:} $\mathcal{K} = \{""" + freqs_str + r"""\}$
  \quad (""" + str(len(circuit.key_frequencies)) + r""" frequencies)

  \textbf{Neurons assigned:} """ + str(len(circuit.neuron_frequency_assignments)) + r"""/512
  \quad (""" + ", ".join(f"$k={k}$: {freq_neuron_counts.get(k, 0)}" for k in circuit.key_frequencies) + r""")
\end{tcolorbox}

\begin{tcolorbox}[colback=green!5!white, colframe=green!75!black, title=The Core Equation]
  \begin{equation}
    \boxed{
      \Logit(c \mid a, b) =
      \sum_{k \in \mathcal{K}}
      \underbrace{\alpha_k}_{\substack{\text{learned} \\ \text{amplitude}}}
      \cdot \underbrace{\cos\!\left(\frac{2\pi k (a + b - c)}{""" + str(P) + r"""}\right)}_{\substack{\text{max at } c = (a+b) \bmod """ + str(P) + r""" \\ \text{(constructive interference)}}}
    }
  \end{equation}
\end{tcolorbox}

\begin{tcolorbox}[colback=orange!5!white, colframe=orange!75!black, title=Data Flow Pipeline]
  \begin{gather*}
    \underbrace{a, b}_{\text{inputs}}
    \xrightarrow{\; W_E \;}
    \underbrace{\cos(\omega_k t),\; \sin(\omega_k t)}_{\text{Fourier embedding}}
    \xrightarrow{\; \attn \;}
    \underbrace{\text{degree-2 products}}_{\substack{\text{attn wt.} \times \text{OV}}} \\[8pt]
    \xrightarrow{\; \MLP \;}
    \underbrace{\cos(\omega_k(a{+}b)),\; \sin(\omega_k(a{+}b))}_{\text{trig addition identities}}
    \xrightarrow{\; W_U \;}
    \underbrace{\cos(\omega_k(a{+}b{-}c))}_{\text{logits}}
  \end{gather*}
\end{tcolorbox}

\tableofcontents
\newpage

"""

        postamble = r"""

\end{document}
"""

        return preamble + body_latex + postamble

    def discover_circuit(progress=gr.Progress()):
        """Run Fourier circuit discovery on the trained model."""
        if state["model"] is None:
            return "No model trained yet!", None, None, None, ""

        logs = []

        def progress_cb(msg):
            logs.append(msg)

        progress(0.1, desc="Starting circuit discovery...")
        discoverer = CircuitDiscoverer(state["model"])
        circuit = discoverer.full_discovery(progress_cb=progress_cb)
        state["circuit"] = circuit

        progress(0.8, desc="Running ablation tests...")
        ablation_results = ablation_test(state["model"], circuit.key_frequencies)
        state["ablation_results"] = ablation_results

        progress(0.9, desc="Generating visualizations...")

        fourier_fig = make_fourier_plot(circuit.embedding_fourier_norms, circuit.wl_fourier_norms, circuit.key_frequencies)
        neuron_fig = make_neuron_assignment_plot(circuit.neuron_frequency_assignments, circuit.key_frequencies, state["model"].d_mlp)
        ablation_fig = make_ablation_plot(ablation_results)

        # Build summary text
        summary = (
            f"## Discovered Circuit\n\n"
            f"**Key Frequencies:** {circuit.key_frequencies}\n\n"
            f"**Mathematical Formula:**\n```\n{circuit.mathematical_formula}\n```\n\n"
            f"**Logit FVE:** {circuit.fve_logits:.4f}\n\n"
            f"**Verification Accuracy:** {circuit.verification_accuracy*100:.2f}%\n\n"
            f"### Algorithm Description\n\n{circuit.algorithm_description}\n\n"
            f"### Ablation Results\n\n"
            f"- Baseline: {ablation_results['baseline_accuracy']:.4f}\n"
            f"- Without key freqs: {ablation_results['accuracy_without_key_freqs']:.4f}\n"
            f"- Only key freqs: {ablation_results['accuracy_restricted_to_key_freqs']:.4f}\n"
        )

        log_text = "\n".join(logs)
        progress(1.0, desc="Done!")

        return summary, fourier_fig, neuron_fig, ablation_fig, log_text

    def run_acdc_discovery(threshold, n_samples, granularity, progress=gr.Progress()):
        """Run ACDC automatic circuit discovery."""
        if state["model"] is None:
            return "No model trained yet! Please train a model first.", None, "", ""

        logs = []

        def progress_cb(msg):
            logs.append(msg)

        progress(0.05, desc="Initializing ACDC...")

        acdc = ACDCCircuitDiscoverer(state["model"], granularity=granularity)

        progress(0.1, desc="Running ACDC algorithm...")
        discovered_circuit = acdc.run_acdc(
            threshold=float(threshold),
            n_samples=int(n_samples),
            progress_cb=progress_cb,
        )

        progress(0.85, desc="Generating summary...")
        summary = acdc.get_circuit_summary(discovered_circuit)
        viz_text = acdc.visualize_circuit(discovered_circuit)

        progress(0.9, desc="Creating visualization...")
        circuit_fig = make_acdc_circuit_plot(summary)

        state["acdc_result"] = {
            "circuit": discovered_circuit,
            "summary": summary,
            "visualization": viz_text,
        }

        # Build markdown summary
        md_summary = (
            f"## ACDC Circuit Discovery Results\n\n"
            f"**Algorithm:** ACDC (Conmy et al., 2023)\n\n"
            f"**Threshold (tau):** {threshold}\n\n"
            f"**Granularity:** {granularity}\n\n"
            f"**Edges:** {summary['num_edges']} / {acdc.graph.num_edges} "
            f"({summary['num_edges']/acdc.graph.num_edges:.1%} retained)\n\n"
            f"**Active Nodes:** {summary['num_active_nodes']} / {len(acdc.graph.nodes)}\n\n"
            f"**Active Attention Heads:** {summary['active_heads']}\n\n"
            f"### Edge Breakdown\n\n"
        )
        for edge_type, count in summary["edge_types"].items():
            if count > 0:
                md_summary += f"- {edge_type}: {count}\n"

        md_summary += f"\n### Circuit Properties\n\n"
        md_summary += f"- MLP pathway active: {'Yes' if summary['has_mlp_path'] else 'No'}\n"
        md_summary += f"- Direct residual pathway: {'Yes' if summary['has_direct_path'] else 'No'}\n"

        log_text = "\n".join(logs)
        progress(1.0, desc="ACDC complete!")

        return md_summary, circuit_fig, viz_text, log_text

    def run_combined_discovery(acdc_threshold, n_samples, progress=gr.Progress()):
        """Run combined Fourier + ACDC discovery."""
        if state["model"] is None:
            return "No model trained yet! Please train a model first.", None, None, None, ""

        logs = []

        def progress_cb(msg):
            logs.append(msg)

        progress(0.05, desc="Starting combined discovery...")

        combined = CombinedCircuitDiscoverer(state["model"])
        results = combined.full_discovery(
            acdc_threshold=float(acdc_threshold),
            n_samples=int(n_samples),
            progress_cb=progress_cb,
        )

        progress(0.9, desc="Generating visualizations...")

        fourier_circuit = results["fourier_circuit"]
        acdc_summary = results["acdc_summary"]

        fourier_fig = make_fourier_plot(
            fourier_circuit.embedding_fourier_norms,
            fourier_circuit.wl_fourier_norms,
            fourier_circuit.key_frequencies,
        )
        acdc_fig = make_acdc_circuit_plot(acdc_summary)

        # Consistency report
        consistency = results["consistency"]
        consistency_md = "## Cross-Validation Results\n\n"
        for check_name, passed in consistency.items():
            icon = "passed" if passed else "FAILED"
            consistency_md += f"- {check_name}: {icon}\n"

        # Combined summary
        md_summary = (
            f"## Combined Circuit Discovery\n\n"
            f"### Fourier Analysis (Nanda et al. 2023)\n\n"
            f"- Key frequencies: {fourier_circuit.key_frequencies}\n"
            f"- Logit FVE: {fourier_circuit.fve_logits:.4f}\n"
            f"- Verification accuracy: {fourier_circuit.verification_accuracy*100:.2f}%\n\n"
            f"### ACDC (Conmy et al. 2023)\n\n"
            f"- Edges retained: {acdc_summary['num_edges']}\n"
            f"- Active heads: {acdc_summary['active_heads']}\n"
            f"- MLP pathway: {'Active' if acdc_summary['has_mlp_path'] else 'Inactive'}\n\n"
            f"{consistency_md}\n\n"
            f"### Interpretation\n\n"
            f"The Fourier analysis identifies WHAT the circuit computes "
            f"(key frequencies {fourier_circuit.key_frequencies}, trig identities), "
            f"while ACDC identifies HOW the circuit is structured "
            f"(which {acdc_summary['num_active_nodes']} components and "
            f"{acdc_summary['num_edges']} connections are essential).\n"
        )

        log_text = "\n".join(logs)
        progress(1.0, desc="Combined discovery complete!")

        return md_summary, fourier_fig, acdc_fig, consistency_md, log_text

    def view_saved_training_data():
        """Load and display saved training metrics and plot."""
        csv_path = os.path.join(SAVE_DIR, "training_metrics.csv")
        if not os.path.exists(csv_path):
            return pd.DataFrame({"message": ["No saved training data found."]}), go.Figure()

        df = pd.read_csv(csv_path)

        # Reconstruct plot from saved data
        train_losses = list(zip(df["epoch"].tolist(), df["train_loss"].tolist()))
        test_accs = list(zip(df["epoch"].tolist(), df["test_acc"].tolist()))
        fig = make_training_plot(train_losses, test_accs)

        return df, fig

    # Build the Gradio app
    with gr.Blocks(title="Grokking Circuit Discovery Tool", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Grokking Circuit Discovery & Verification Tool")
        gr.Markdown(
            "Train a small transformer on modular addition, observe grokking, "
            "and automatically discover the Fourier multiplication circuit using "
            "both Fourier analysis (Nanda et al. 2023) and ACDC (Conmy et al. 2023)."
        )

        with gr.Tabs():
            # ===== TAB 1: Training =====
            with gr.TabItem("Training"):
                gr.Markdown("### Model & Training Configuration")
                with gr.Row():
                    with gr.Column(scale=1):
                        p_input = gr.Number(value=113, label="Prime P (modulus)", precision=0)
                        d_model_input = gr.Number(value=128, label="d_model", precision=0)
                        n_heads_input = gr.Number(value=4, label="n_heads", precision=0)
                        d_mlp_input = gr.Number(value=512, label="d_mlp", precision=0)
                    with gr.Column(scale=1):
                        train_frac_input = gr.Number(value=0.3, label="Train fraction")
                        epochs_input = gr.Number(value=80000, label="Max epochs", precision=0)
                        lr_input = gr.Number(value=1e-3, label="Learning rate")
                        wd_input = gr.Number(value=1.0, label="Weight decay")

                train_btn = gr.Button("Train Model", variant="primary", size="lg")

                gr.Markdown("### Training Progress")
                training_plot = gr.Plot(label="Training Curves (Live)")
                training_table = gr.Dataframe(label="Training Metrics Table", interactive=False)
                training_log = gr.Textbox(label="Training Log", lines=10, interactive=False)

                train_btn.click(
                    fn=train_and_update,
                    inputs=[p_input, d_model_input, n_heads_input, d_mlp_input,
                            train_frac_input, epochs_input, lr_input, wd_input],
                    outputs=[training_plot, training_table, training_log],
                )

            # ===== TAB 2: Fourier Circuit Discovery =====
            with gr.TabItem("Fourier Discovery"):
                gr.Markdown("### Discover the Fourier Multiplication Circuit (Nanda et al. 2023)")
                gr.Markdown(
                    "Analyzes the trained model's weights in Fourier space to identify "
                    "key frequencies, verify trig identities in the MLP, and test the "
                    "discovered formula exhaustively on all P*P inputs."
                )
                discover_btn = gr.Button("Discover Circuit", variant="primary", size="lg")

                circuit_summary = gr.Markdown(label="Circuit Summary")
                with gr.Row():
                    fourier_plot = gr.Plot(label="Fourier Analysis")
                    neuron_plot = gr.Plot(label="Neuron Assignments")
                ablation_plot = gr.Plot(label="Ablation Results")
                discovery_log = gr.Textbox(label="Discovery Log", lines=15, interactive=False)

                discover_btn.click(
                    fn=discover_circuit,
                    inputs=[],
                    outputs=[circuit_summary, fourier_plot, neuron_plot, ablation_plot, discovery_log],
                )

            # ===== TAB 3: ACDC Circuit Discovery =====
            with gr.TabItem("ACDC Discovery"):
                gr.Markdown("### Automatic Circuit Discovery (Conmy et al. 2023)")
                gr.Markdown(
                    "Implements the ACDC algorithm which iterates from outputs to inputs "
                    "through the computational graph, attempting to remove edges that don't "
                    "significantly affect model performance (measured by KL divergence). "
                    "This finds the minimal subgraph that implements the modular addition behavior."
                )

                with gr.Row():
                    acdc_threshold = gr.Number(value=0.01, label="Threshold (tau)", info="Edges causing KL increase below this are removed")
                    acdc_n_samples = gr.Number(value=512, label="N samples", precision=0, info="Number of data points for patching")
                    acdc_granularity = gr.Dropdown(
                        choices=["component", "fine", "neuron"],
                        value="component",
                        label="Granularity",
                        info="Level of detail for computational graph",
                    )

                acdc_btn = gr.Button("Run ACDC", variant="primary", size="lg")

                acdc_summary_md = gr.Markdown(label="ACDC Summary")
                acdc_circuit_plot = gr.Plot(label="Circuit Diagram")
                acdc_viz_text = gr.Textbox(label="Circuit Visualization (Text)", lines=20, interactive=False)
                acdc_log = gr.Textbox(label="ACDC Log", lines=15, interactive=False)

                acdc_btn.click(
                    fn=run_acdc_discovery,
                    inputs=[acdc_threshold, acdc_n_samples, acdc_granularity],
                    outputs=[acdc_summary_md, acdc_circuit_plot, acdc_viz_text, acdc_log],
                )

            # ===== TAB 4: Combined Discovery =====
            with gr.TabItem("Combined Discovery"):
                gr.Markdown("### Combined Fourier + ACDC Analysis")
                gr.Markdown(
                    "Runs both Fourier analysis and ACDC, then cross-validates the results. "
                    "Fourier analysis identifies WHAT the circuit computes (key frequencies, "
                    "trig identities), while ACDC identifies HOW the circuit is structured "
                    "(which components and connections are essential)."
                )

                with gr.Row():
                    combined_threshold = gr.Number(value=0.01, label="ACDC Threshold (tau)")
                    combined_n_samples = gr.Number(value=512, label="N samples", precision=0)

                combined_btn = gr.Button("Run Combined Discovery", variant="primary", size="lg")

                combined_summary_md = gr.Markdown(label="Combined Summary")
                with gr.Row():
                    combined_fourier_plot = gr.Plot(label="Fourier Analysis")
                    combined_acdc_plot = gr.Plot(label="ACDC Circuit")
                combined_consistency_md = gr.Markdown(label="Cross-Validation")
                combined_log = gr.Textbox(label="Combined Log", lines=15, interactive=False)

                combined_btn.click(
                    fn=run_combined_discovery,
                    inputs=[combined_threshold, combined_n_samples],
                    outputs=[combined_summary_md, combined_fourier_plot, combined_acdc_plot, combined_consistency_md, combined_log],
                )

            with gr.TabItem("Saved Runs"):
                gr.Markdown("### Load a Previous Training Run")
                refresh_btn = gr.Button("Refresh Run List")
                run_dropdown = gr.Dropdown(choices=[], label="Select a run")
                load_run_btn = gr.Button("Load Selected Run", variant="primary")
                loaded_info = gr.Markdown()
                loaded_plot = gr.Plot()
                loaded_table = gr.Dataframe()

                def refresh_runs():
                    """Refresh the dropdown with available runs."""
                    runs = list_saved_runs()
                    choices = [r["run_id"] for r in runs]
                    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

                def load_selected_run(run_id):
                    """Load a run and display its data."""
                    if not run_id:
                        return "No run selected.", go.Figure(), pd.DataFrame()

                    model, train_losses, test_accs, metrics_df, config = load_run(run_id)
                    state["model"] = model
                    state["train_losses"] = train_losses
                    state["test_accs"] = test_accs

                    fig = make_training_plot(train_losses, test_accs)

                    info_md = (
                        f"**Run ID:** {run_id}\n\n"
                        f"**Config:** P={config['P']}, d_model={config['d_model']}, "
                        f"n_heads={config['n_heads']}, d_mlp={config['d_mlp']}\n\n"
                        f"**Train frac:** {config['train_frac']}, **LR:** {config['lr']}, "
                        f"**WD:** {config['weight_decay']}\n\n"
                        f"**Final test acc:** {metrics_df['test_acc'].iloc[-1]:.4f}\n\n"
                        f"✅ Model loaded into state — you can now run circuit discovery."
                    )

                    return info_md, fig, metrics_df

                refresh_btn.click(
                    fn=refresh_runs,
                    inputs=[],
                    outputs=[run_dropdown],
                )

                load_run_btn.click(
                    fn=load_selected_run,
                    inputs=[run_dropdown],
                    outputs=[loaded_info, loaded_plot, loaded_table],
                )

            # ===== TAB 6: Interactive Inference =====
            with gr.TabItem("Run Inference"):
                gr.Markdown("### Enter Numbers to Test the Neural Network")
                gr.Markdown(
                    "Enter two numbers `a` and `b` (between 0 and P-1) to see what the trained model "
                    "predicts for `(a + b) mod P`. You can also enter multiple pairs to test in batch."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        input_a = gr.Number(value=7, label="Input a", precision=0, info="Integer from 0 to P-1")
                        input_b = gr.Number(value=13, label="Input b", precision=0, info="Integer from 0 to P-1")
                        infer_btn = gr.Button("Run Prediction", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        infer_result_md = gr.Markdown(label="Prediction Result")

                gr.Markdown("---")
                gr.Markdown("### Batch Inference")
                gr.Markdown(
                    "Enter comma-separated pairs (e.g., `3,5; 10,20; 50,63`) to test multiple inputs at once."
                )
                batch_input = gr.Textbox(
                    label="Batch input (format: a1,b1; a2,b2; ...)",
                    placeholder="3,5; 10,20; 50,63; 100,12",
                    lines=2,
                )
                batch_btn = gr.Button("Run Batch Prediction", variant="secondary")
                batch_result_table = gr.Dataframe(label="Batch Results", interactive=False)
                batch_top_k_plot = gr.Plot(label="Top-K Logits for Last Input")

                def run_single_inference(a, b):
                    """Run the model on a single (a, b) pair."""
                    if state["model"] is None:
                        return "⚠️ **No model trained yet!** Please go to the Training tab and train a model first."

                    model = state["model"]
                    P = model.P
                    a_int = int(a) % P
                    b_int = int(b) % P
                    correct_answer = (a_int + b_int) % P

                    a_tensor = torch.tensor([a_int])
                    b_tensor = torch.tensor([b_int])

                    with torch.no_grad():
                        logits = model(a_tensor, b_tensor)  # (1, P)
                        probs = F.softmax(logits, dim=-1)
                        predicted = logits.argmax(dim=-1).item()
                        confidence = probs[0, predicted].item()

                    # Top 5 predictions
                    top_k_values, top_k_indices = torch.topk(probs[0], k=min(5, P))

                    is_correct = "✅" if predicted == correct_answer else "❌"

                    result = (
                        f"## Result\n\n"
                        f"**Input:** a = {a_int}, b = {b_int}\n\n"
                        f"**True answer:** ({a_int} + {b_int}) mod {P} = **{correct_answer}**\n\n"
                        f"**Model prediction:** **{predicted}** (confidence: {confidence*100:.2f}%) {is_correct}\n\n"
                        f"### Top 5 Predictions\n\n"
                        f"| Rank | Value | Probability |\n"
                        f"|------|-------|-------------|\n"
                    )
                    for i in range(len(top_k_indices)):
                        val = top_k_indices[i].item()
                        prob = top_k_values[i].item()
                        marker = " ← correct" if val == correct_answer else ""
                        result += f"| {i+1} | {val} | {prob*100:.3f}%{marker} |\n"

                    return result

                def run_batch_inference(batch_str):
                    """Run the model on multiple (a, b) pairs."""
                    if state["model"] is None:
                        return pd.DataFrame({"error": ["No model trained yet!"]}), go.Figure()

                    model = state["model"]
                    P = model.P

                    # Parse input
                    pairs = []
                    try:
                        for pair_str in batch_str.strip().split(";"):
                            pair_str = pair_str.strip()
                            if not pair_str:
                                continue
                            parts = pair_str.split(",")
                            a_val = int(parts[0].strip()) % P
                            b_val = int(parts[1].strip()) % P
                            pairs.append((a_val, b_val))
                    except (ValueError, IndexError):
                        return pd.DataFrame({"error": ["Invalid input format. Use: a1,b1; a2,b2; ..."]}), go.Figure()

                    if not pairs:
                        return pd.DataFrame({"error": ["No valid pairs found."]}), go.Figure()

                    # Run inference
                    a_tensor = torch.tensor([p[0] for p in pairs])
                    b_tensor = torch.tensor([p[1] for p in pairs])

                    with torch.no_grad():
                        logits = model(a_tensor, b_tensor)
                        probs = F.softmax(logits, dim=-1)
                        predictions = logits.argmax(dim=-1)

                    # Build results table
                    rows = []
                    for i, (a_val, b_val) in enumerate(pairs):
                        correct = (a_val + b_val) % P
                        pred = predictions[i].item()
                        conf = probs[i, pred].item()
                        rows.append({
                            "a": a_val,
                            "b": b_val,
                            "correct ((a+b) mod P)": correct,
                            "predicted": pred,
                            "confidence (%)": round(conf * 100, 2),
                            "correct?": "✅" if pred == correct else "❌",
                        })

                    df = pd.DataFrame(rows)

                    # Plot top-K logits for the last input
                    last_probs = probs[-1].cpu().numpy()
                    top_k = min(10, P)
                    top_indices = np.argsort(last_probs)[-top_k:][::-1]
                    top_probs = last_probs[top_indices]

                    last_correct = (pairs[-1][0] + pairs[-1][1]) % P
                    colors = ["green" if idx == last_correct else "steelblue" for idx in top_indices]

                    fig = go.Figure(data=[
                        go.Bar(
                            x=[str(idx) for idx in top_indices],
                            y=top_probs,
                            marker_color=colors,
                        )
                    ])
                    fig.update_layout(
                        title=f"Top-{top_k} Predictions for a={pairs[-1][0]}, b={pairs[-1][1]} (green = correct: {last_correct})",
                        xaxis_title="Output class",
                        yaxis_title="Probability",
                        height=400,
                    )

                    return df, fig

                infer_btn.click(
                    fn=run_single_inference,
                    inputs=[input_a, input_b],
                    outputs=[infer_result_md],
                )

                batch_btn.click(
                    fn=run_batch_inference,
                    inputs=[batch_input],
                    outputs=[batch_result_table, batch_top_k_plot],
                )

            # ===== TAB: LaTeX Equations =====
            with gr.TabItem("📐 LaTeX Equations"):
                gr.Markdown("### Full LaTeX Equations for the Discovered Circuit")
                gr.Markdown(
                    "After running **Fourier Discovery**, click below to generate the complete "
                    "set of LaTeX equations with `\\underbrace` and `\\overbrace` annotations "
                    "showing exactly where each piece of data comes from. "
                    "You can copy the raw LaTeX into any `.tex` file to compile a beautiful PDF, "
                    "or view the rendered equations directly below."
                )

                latex_btn = gr.Button(
                    "🧮 Generate Full LaTeX Equations",
                    variant="primary",
                    size="lg",
                )

                gr.Markdown("---")
                gr.Markdown("#### Rendered Equations (in-app preview)")
                latex_display_md = gr.Markdown(
                    value="*Run Fourier Discovery first, then click the button above.*",
                    label="Rendered Equations",
                )

                gr.Markdown("---")
                gr.Markdown("#### Raw LaTeX (copy into a .tex file)")
                gr.Markdown(
                    "Copy this into a LaTeX document with `\\usepackage{amsmath}` and "
                    "`\\usepackage{tcolorbox}` to compile."
                )
                latex_raw_output = gr.Textbox(
                    label="Full LaTeX Source",
                    lines=30,
                    interactive=False
                )

                gr.Markdown("---")
                gr.Markdown("#### Quick Reference Card (LaTeX)")
                latex_quick_ref = gr.Textbox(
                    label="Quick Reference LaTeX",
                    lines=15,
                    interactive=False
                )

                latex_btn.click(
                    fn=render_latex_to_display,
                    inputs=[],
                    outputs=[latex_display_md, latex_raw_output, latex_quick_ref],
                )

            # ===== TAB: Live Activation Viewer =====
            with gr.TabItem("🔬 Live Activations"):
                gr.Markdown("### Live Layer-by-Layer Activation Viewer")
                gr.Markdown(
                    "Enter `a` and `b` to see what comes out of **every** component in the network: "
                    "embeddings, each attention head, the dense MLP layers (pre/post ReLU), "
                    "residual streams, and final logits — all with real values."
                )

                with gr.Row():
                    live_a = gr.Number(value=7, label="Input a", precision=0)
                    live_b = gr.Number(value=13, label="Input b", precision=0)
                    live_btn = gr.Button("🔍 Run & Visualize All Layers", variant="primary")

                gr.Markdown("---")
                gr.Markdown("#### Embedding Space — Auto-Discovered Circles")
                gr.Markdown(
                    "Click **Auto-Find Circles** to automatically discover which projections "
                    "of the embedding space show circular structure. Uses Fourier projection "
                    "(if circuit is discovered), PCA, and raw dimension scanning — no more "
                    "manual dimension hunting!"
                )

                auto_circle_btn = gr.Button(
                    "🔍 Auto-Find All Circles", variant="primary"
                )
                all_circles_plot = gr.Plot(label="All Discovered Circles (Summary Grid)")

                gr.Markdown("#### Inspect Individual Circle")
                circle_dropdown = gr.Dropdown(
                    choices=[], label="Select a discovered circle to inspect",
                    interactive=True,
                )
                inspect_circle_btn = gr.Button("Inspect Selected Circle")
                single_circle_plot = gr.Plot(label="Detailed Circle View")

                # State for discovered pairs
                discovered_pairs_state = gr.State([])

                gr.Markdown("---")
                gr.Markdown("#### Manual Override (if needed)")
                with gr.Row():
                    dim_x_select = gr.Number(value=0, label="X Dimension", precision=0)
                    dim_y_select = gr.Number(value=1, label="Y Dimension", precision=0)
                    token_select = gr.Number(value=0, label="Token ID (for all-dims view)", precision=0)

                with gr.Row():
                    embed_circle_btn = gr.Button("Show Manual Pair")
                    embed_alldims_btn = gr.Button("Show All Dimensions for Token")

                embed_circle_plot = gr.Plot(label="Embedding Circle (2D projection)")
                embed_alldims_plot = gr.Plot(label="All Embedding Dimensions")

                gr.Markdown("---")
                gr.Markdown("#### Layer Outputs (Live)")

                embed_plot = gr.Plot(label="Embeddings (Token + Positional + Combined)")
                attn_weights_plot = gr.Plot(label="Attention Weights")
                attn_heads_plot = gr.Plot(label="Per-Head Attention Outputs")
                attn_combined_plot = gr.Plot(label="Combined Attention Output (after W_O)")
                residual_mid_plot = gr.Plot(label="Residual Stream (mid)")
                mlp_pre_plot = gr.Plot(label="MLP Pre-Activations (before ReLU)")
                mlp_hidden_plot = gr.Plot(label="MLP Hidden (after ReLU)")
                mlp_out_plot = gr.Plot(label="MLP Output")
                residual_final_plot = gr.Plot(label="Final Residual Stream")
                logits_plot = gr.Plot(label="Output Logits")

                def run_live_activations(a, b):
                    if state["model"] is None:
                        empty = go.Figure().update_layout(
                            title="⚠ No model loaded! Train or load a model first.",
                            annotations=[dict(
                                text="No model available.<br>Go to Training tab or Saved Runs tab first.",
                                xref="paper", yref="paper", x=0.5, y=0.5,
                                showarrow=False, font=dict(size=16, color="red")
                            )]
                        )
                        return [empty] * 10

                    try:
                        figs = make_layer_activation_plots(state["model"], int(a), int(b))
                        return [
                            figs["embeddings"],
                            figs["attention_weights"],
                            figs["attn_head_outputs"],
                            figs["attn_combined_output"],
                            figs["residual_mid"],
                            figs["mlp_pre"],
                            figs["mlp_hidden"],
                            figs["mlp_output"],
                            figs["residual_final"],
                            figs["logits"],
                        ]
                    except Exception as e:
                        error_fig = go.Figure().update_layout(
                            title=f"❌ Error: {str(e)}",
                        )
                        return [error_fig] * 10

                def auto_find_circles():
                    if state["model"] is None:
                        empty_fig = go.Figure().update_layout(
                            title="⚠ No model loaded! Train or load a model first.",
                        )
                        return empty_fig, gr.Dropdown(choices=[]), []

                    key_freqs = None
                    if state.get("circuit") and state["circuit"] is not None:
                        key_freqs = state["circuit"].key_frequencies

                    pairs = find_circular_dimension_pairs(
                        state["model"], key_freqs, top_k_pairs=5
                    )

                    summary_fig = make_all_circles_summary(state["model"], key_freqs)

                    # Build dropdown choices
                    choices = [
                        f"{i}: {p['description']} (score={p['circularity_score']:.3f})"
                        for i, p in enumerate(pairs)
                    ]

                    return (
                        summary_fig,
                        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
                        pairs,
                    )

                def inspect_selected_circle(selection, pairs):
                    if state["model"] is None or not pairs or not selection:
                        return go.Figure()

                    # Parse index from selection string
                    try:
                        idx = int(selection.split(":")[0])
                    except (ValueError, IndexError):
                        return go.Figure()

                    if idx >= len(pairs):
                        return go.Figure()

                    return make_auto_circle_plot(state["model"], pairs[idx])

                def show_embed_circle(dim_x, dim_y):
                    if state["model"] is None:
                        return go.Figure()
                    return make_embedding_circle_plot(state["model"], int(dim_x), int(dim_y))

                def show_embed_alldims(token_id):
                    if state["model"] is None:
                        return go.Figure()
                    return make_embedding_all_dims_heatmap(state["model"], int(token_id))

                live_btn.click(
                    fn=run_live_activations,
                    inputs=[live_a, live_b],
                    outputs=[embed_plot, attn_weights_plot, attn_heads_plot,
                             attn_combined_plot, residual_mid_plot, mlp_pre_plot,
                             mlp_hidden_plot, mlp_out_plot, residual_final_plot, logits_plot],
                )

                auto_circle_btn.click(
                    fn=auto_find_circles,
                    inputs=[],
                    outputs=[all_circles_plot, circle_dropdown, discovered_pairs_state],
                )

                inspect_circle_btn.click(
                    fn=inspect_selected_circle,
                    inputs=[circle_dropdown, discovered_pairs_state],
                    outputs=[single_circle_plot],
                )

                embed_circle_btn.click(
                    fn=show_embed_circle,
                    inputs=[dim_x_select, dim_y_select],
                    outputs=[embed_circle_plot],
                )

                embed_alldims_btn.click(
                    fn=show_embed_alldims,
                    inputs=[token_select],
                    outputs=[embed_alldims_plot],
                )

            # ===== TAB 7: About =====
            with gr.TabItem("About"):
                gr.Markdown("""
                ## About This Tool

                This tool implements circuit discovery methodologies from two papers:

                ### 1. Fourier Analysis (Nanda et al., 2023)
                **"Progress Measures for Grokking via Mechanistic Interpretability"**

                Discovers the Fourier multiplication algorithm by analyzing model weights
                in Fourier space. The algorithm:
                1. **Embedding**: Maps inputs to Fourier components (sin/cos at key frequencies)
                2. **Attention**: Moves information from input positions to the output position
                3. **MLP**: Computes trig identities to get cos(wk*(a+b)) from cos(wk*a), sin(wk*a), etc.
                4. **Unembedding**: Converts back from Fourier space to logits via constructive interference

                ### 2. ACDC (Conmy et al., 2023)
                **"Towards Automated Circuit Discovery for Mechanistic Interpretability"**

                Automatically finds the minimal computational subgraph (circuit) that implements
                a behavior. The ACDC algorithm:
                1. Starts with the full computational graph
                2. Iterates from outputs to inputs (reverse topological order)
                3. At each node, tests whether each incoming edge can be removed
                4. Removes edges whose removal causes KL divergence increase below threshold tau
                5. Returns the pruned subgraph as the discovered circuit

                ### Key Metrics
                - **FVE (Fraction of Variance Explained)**: How well the discovered formula explains model behavior
                - **KL Divergence**: Measures how much the circuit's output differs from the full model
                - **Ablation accuracy**: Confirms key frequencies are necessary and sufficient
                - **Exhaustive verification**: Tests the formula on all P squared possible inputs
                - **Compression ratio**: How much smaller the circuit is vs. the full graph

                ### What is Grokking?

                Grokking is a phenomenon where a neural network first memorizes training data
                (achieving high train accuracy but low test accuracy), then suddenly generalizes
                after many more training steps. This tool lets you observe grokking in real-time
                and then reverse-engineer the algorithm the model learns.
                """)

    return demo

def save_run(model, train_losses, test_accs, metrics_table, config: dict):
    """Save a complete training run: model weights + metrics + config."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(SAVE_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Save config (hyperparameters needed to reconstruct the model)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save metrics
    df = pd.DataFrame(metrics_table)
    df.to_csv(run_dir / "metrics.csv", index=False)

    # Save loss/acc curves
    with open(run_dir / "curves.json", "w") as f:
        json.dump({"train_losses": train_losses, "test_accs": test_accs}, f)

    return run_id

def list_saved_runs() -> list[dict]:
    """List all saved runs with their summary info."""
    runs = []
    save_path = Path(SAVE_DIR)
    if not save_path.exists():
        return runs

    for run_dir in sorted(save_path.iterdir(), reverse=True):
        if run_dir.is_dir() and (run_dir / "config.json").exists():
            with open(run_dir / "config.json") as f:
                config = json.load(f)
            # Get final test accuracy from metrics
            metrics_path = run_dir / "metrics.csv"
            final_acc = "N/A"
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                if len(df) > 0:
                    final_acc = f"{df['test_acc'].iloc[-1]:.4f}"
            runs.append({
                "run_id": run_dir.name,
                "P": config.get("P"),
                "epochs_trained": config.get("epochs"),
                "final_test_acc": final_acc,
            })
    return runs


def load_run(run_id: str):
    """Load a saved run: reconstruct model and return metrics."""
    run_dir = Path(SAVE_DIR) / run_id

    with open(run_dir / "config.json") as f:
        config = json.load(f)

    # Reconstruct model architecture from config
    model = ModularAdditionTransformer(
        P=config["P"], d_model=config["d_model"],
        n_heads=config["n_heads"], d_mlp=config["d_mlp"]
    )
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu"))
    model.eval()

    with open(run_dir / "curves.json") as f:
        curves = json.load(f)

    metrics_df = pd.read_csv(run_dir / "metrics.csv")

    return model, curves["train_losses"], curves["test_accs"], metrics_df, config

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    demo = build_gui()
    try:
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
    except OSError as e:
        print(e)
        sys.exit(1)
