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

DEFAULT_P = 113

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
import gradio as gr

from circuit_equation_viewer import build_equation_viewer_tab

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

# Zyklische Colorscale: Blau → Grün → Blau (betont Zyklizität)
CYCLIC_BLUE_GREEN = [
    [0.0,  "rgb(0, 50, 200)"],     # Blau (Start)
    [0.5, "rgb(0, 150, 150)"],    # Übergang Blau→Grün
    [1.0,  "rgb(0, 200, 80)"],     # Grün (Mitte)
]


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
        W_E_fourier = fourier_basis @ W_E  # (P, d_model)

        for k in key_frequencies:
            # The cos and sin rows for frequency k
            cos_row = W_E_fourier[2*k - 1]  # (d_model,)
            sin_row = W_E_fourier[2*k] if 2*k < P else np.zeros(d_model)

            # Project all token embeddings onto these two directions
            x_proj = W_E @ cos_row / (np.linalg.norm(cos_row)**2 + 1e-10)  # (P,)
            y_proj = W_E @ sin_row / (np.linalg.norm(sin_row)**2 + 1e-10)  # (P,)

            # Compute circularity score
            cx, cy = x_proj.mean(), y_proj.mean()
            radii = np.sqrt((x_proj - cx)**2 + (y_proj - cy)**2)
            mean_r = radii.mean()
            circularity = 1.0 - (radii.std() / (mean_r + 1e-10))

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
    from scipy.linalg import svd
    W_centered = W_E - W_E.mean(axis=0, keepdims=True)
    U, S, Vt = svd(W_centered, full_matrices=False)

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
                "frequency": None,
                "circularity_score": float(combined_score),
                "x_coords": x_proj,
                "y_coords": y_proj,
                "description": f"PCA pair (PC{i}, PC{j}): circularity={combined_score:.3f}",
                "method": "pca",
            })

    pair_scores.sort(key=lambda x: -x["circularity_score"])
    results.extend(pair_scores[:top_k_pairs])

    # === Method 3: Raw dimension pairs (fast scan) ===
    if d_model <= 32:
        raw_pairs = [(i, j) for i in range(d_model) for j in range(i+1, d_model)]
    else:
        variances = W_E.var(axis=0)
        top_dims = np.argsort(variances)[-20:]
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
    Uses a cyclic blue→green→blue colorscale to emphasize circular ordering.
    Color is based on token NUMBER (0..P-1), NOT geometric angle.
    """
    P = model.P
    x_coords = pair_info["x_coords"]
    y_coords = pair_info["y_coords"]

    fig = go.Figure()

    # Color based on token number (cyclic: 0 → blue, P//2 → green, P-1 → almost blue again)
    color_values = np.arange(P) / P  # [0, 1), cyclic

    # Still compute angles for hover info
    cx, cy = x_coords.mean(), y_coords.mean()
    angles = np.arctan2(y_coords - cy, x_coords - cx)  # [-pi, pi]

    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers+text",
        marker=dict(
            size=10,
            color=color_values,
            colorscale=CYCLIC_BLUE_GREEN,
            showscale=True,
            colorbar=dict(
                title="Token-Nummer",
                tickvals=[0, 0.25, 0.5, 0.75],
                ticktext=[f"0", f"{P//4}", f"{P//2}", f"{3*P//4}"],
            ),
        ),
        text=[str(i) for i in range(P)],
        textposition="top center",
        textfont=dict(size=7),
        hovertemplate=(
            "Token: %{text}<br>"
            "x: %{x:.4f}<br>"
            "y: %{y:.4f}<br>"
            "Winkel: %{customdata[0]:.1f}°<br>"
            "<extra></extra>"
        ),
        customdata=np.stack([np.degrees(angles) % 360], axis=-1),
        name="Tokens",
    ))

    # Fit circle
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
            title=f"{pair_info['description']}<br>x: {pair_info['dim_x']}, y: {pair_info['dim_y']}<br>Circularity: {circularity:.4f}{freq_str}",
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
    Uses cyclic blue→green→blue colorscale based on token NUMBER.
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
    # Color based on token number, same for all subplots
    color_values = np.arange(P) / P  # [0, 1), cyclic

    for idx, pair in enumerate(pairs[:n_plots]):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        x_coords = pair["x_coords"]
        y_coords = pair["y_coords"]

        fig.add_trace(go.Scatter(
            x=x_coords, y=y_coords,
            mode="markers",
            marker=dict(size=4, color=color_values, colorscale=CYCLIC_BLUE_GREEN, showscale=False),
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
# Winkel-Analyse Tool
# =============================================================================

def compute_circle_angles(model: ModularAdditionTransformer, pair_info: dict) -> tuple:
    """
    Berechnet die Winkel aller Tokens auf dem Kreis und gibt zurück:
    1. Eine Tabelle mit den absoluten Winkeln jedes Tokens
    2. Eine Tabelle mit den paarweisen Winkeldifferenzen (alle mit allen)
    3. Eine Plotly-Figur, die die Winkel visuell auf dem Kreis zeigt
    4. Eine Heatmap der paarweisen Differenzen

    Farben basieren auf der TOKEN-NUMMER, nicht auf dem geometrischen Winkel.

    Returns: (angle_table: pd.DataFrame, diff_df: pd.DataFrame, angle_fig: go.Figure, heatmap_fig: go.Figure)
    """
    P = model.P
    x_coords = pair_info["x_coords"]
    y_coords = pair_info["y_coords"]

    # Mittelpunkt berechnen
    cx, cy = x_coords.mean(), y_coords.mean()

    # Winkel relativ zum Mittelpunkt (in Grad)
    angles_rad = np.arctan2(y_coords - cy, x_coords - cx)
    angles_deg = np.degrees(angles_rad) % 360  # Normalisiere auf [0, 360)

    # === Tabelle 1: Absolute Winkel ===
    radii = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    angle_table = pd.DataFrame({
        "Token": np.arange(P),
        "Winkel (°)": np.round(angles_deg, 2),
        "Winkel (rad)": np.round(angles_rad, 4),
        "x": np.round(x_coords, 4),
        "y": np.round(y_coords, 4),
        "Radius": np.round(radii, 4),
    })

    # === Tabelle 2: Paarweise Winkeldifferenzen (alle mit allen) ===
    diff_matrix = np.zeros((P, P))
    for i in range(P):
        for j in range(P):
            diff = (angles_deg[j] - angles_deg[i]) % 360
            # Normalisiere auf [-180, 180] für kürzesten Winkel
            if diff > 180:
                diff -= 360
            diff_matrix[i, j] = diff

    diff_df = pd.DataFrame(
        np.round(diff_matrix, 2),
        index=[f"{i}" for i in range(P)],
        columns=[f"{j}" for j in range(P)],
    )

    # === Visualisierung: Winkel auf dem Kreis ===
    avg_r = radii.mean()
    fig = go.Figure()

    # Kreis zeichnen
    theta_circle = np.linspace(0, 2*np.pi, 200)
    fig.add_trace(go.Scatter(
        x=cx + avg_r * np.cos(theta_circle),
        y=cy + avg_r * np.sin(theta_circle),
        mode="lines",
        line=dict(color="rgba(200,200,200,0.5)", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Tokens mit Farbe nach TOKEN-NUMMER (zyklisch), NICHT nach Winkel
    color_values = np.arange(P) / P  # [0, 1), cyclic

    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers+text",
        marker=dict(
            size=10,
            color=color_values,
            colorscale=CYCLIC_BLUE_GREEN,
            showscale=True,
            colorbar=dict(
                title="Token-Nummer",
                tickvals=[0, 0.25, 0.5, 0.75],
                ticktext=[f"0", f"{P//4}", f"{P//2}", f"{3*P//4}"],
            ),
        ),
        text=[str(i) for i in range(P)],
        textposition="top center",
        textfont=dict(size=7),
        hovertemplate=(
            "Token: %{text}<br>"
            "Winkel: %{customdata[0]:.1f}°<br>"
            "Radius: %{customdata[1]:.4f}<br>"
            "x: %{x:.4f}<br>"
            "y: %{y:.4f}<br>"
            "<extra></extra>"
        ),
        customdata=np.stack([angles_deg, radii], axis=-1),
        name="Tokens",
    ))

    # Winkellinien vom Mittelpunkt zu ausgewählten Tokens
    step = max(1, P // 12)
    for i in range(0, P, step):
        fig.add_trace(go.Scatter(
            x=[cx, x_coords[i]],
            y=[cy, y_coords[i]],
            mode="lines",
            line=dict(color="rgba(100,100,100,0.3)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Winkel-Annotation auf halber Strecke
        mid_x = cx + 0.55 * (x_coords[i] - cx)
        mid_y = cy + 0.55 * (y_coords[i] - cy)
        fig.add_annotation(
            x=mid_x, y=mid_y,
            text=f"{angles_deg[i]:.0f}°",
            showarrow=False,
            font=dict(size=8, color="gray"),
        )

    # Winkelbögen zwischen aufeinanderfolgenden markierten Tokens
    for i in range(0, P, step):
        j = (i + step) % P
        angle_start = angles_rad[i]
        angle_end = angles_rad[j]
        # Sicherstellen dass wir den kurzen Bogen nehmen
        diff = (angle_end - angle_start) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        arc_angles = np.linspace(angle_start, angle_start + diff, 20)
        arc_r = avg_r * 0.35  # Innerer Bogen
        fig.add_trace(go.Scatter(
            x=cx + arc_r * np.cos(arc_angles),
            y=cy + arc_r * np.sin(arc_angles),
            mode="lines",
            line=dict(color="rgba(255,100,0,0.4)", width=1.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    freq_str = f" (Frequenz k={pair_info.get('frequency', '?')})" if pair_info.get("frequency") else ""
    expected_step = 360.0 / P
    fig.update_layout(
        title=(
            f"Winkelverteilung auf dem Kreis{freq_str}<br>"
            f"<sub>Erwarteter Winkelschritt: {expected_step:.2f}° pro Token | "
            f"Mittelpunkt: ({cx:.3f}, {cy:.3f})</sub>"
        ),
        xaxis=dict(scaleanchor="y", scaleratio=1),
        height=700,
        width=700,
    )

    # === Heatmap der paarweisen Differenzen ===
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=diff_matrix,
        x=list(range(P)),
        y=list(range(P)),
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="Δ Winkel (°)"),
        hovertemplate="Token %{y} → Token %{x}: %{z:.1f}°<extra></extra>",
    ))
    fig_heatmap.update_layout(
        title=f"Paarweise Winkeldifferenzen (alle {P}×{P} Paare){freq_str}",
        xaxis_title="Token j",
        yaxis_title="Token i",
        height=700,
        width=750,
    )

    return angle_table, diff_df, fig, fig_heatmap

# =============================================================================
# Training with Live Progress
# =============================================================================

def train_model(P: int = DEFAULT_P, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512,
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

from acc_circuit_discoverer import ACDCCircuitDiscoverer

# =============================================================================
# Fourier Circuit Discovery (Nanda et al. 2023)
# =============================================================================

from discovered_circuit import DiscoveredCircuit
from circuit_discoverer import CircuitDiscoverer

# =============================================================================
# Combined Circuit Discovery: Fourier + ACDC
# =============================================================================

from combined_circuit_discoverer import CombinedCircuitDiscoverer

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
    Color is based on token NUMBER (0..P-1), NOT geometric angle.
    """
    P = model.P
    W_E = model.embed.weight[:P].detach().cpu().numpy()  # (P, d_model)

    x_coords = W_E[:, dim_x]
    y_coords = W_E[:, dim_y]

    fig = go.Figure()

    # Color based on token number (cyclic), NOT on geometric angle
    color_values = np.arange(P) / P  # [0, 1), cyclic

    # Still compute angles for hover info
    cx, cy = x_coords.mean(), y_coords.mean()
    angles = np.arctan2(y_coords - cy, x_coords - cx)  # [-pi, pi]

    # Plot all token positions with cyclic blue→green→blue colorscale
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="markers+text",
        marker=dict(
            size=10,
            color=color_values,
            colorscale=CYCLIC_BLUE_GREEN,
            showscale=True,
            colorbar=dict(
                title="Token-Nummer",
                tickvals=[0, 0.25, 0.5, 0.75],
                ticktext=[f"0", f"{P//4}", f"{P//2}", f"{3*P//4}"],
            ),
        ),
        text=[str(i) for i in range(P)],
        textposition="top center",
        textfont=dict(size=7),
        customdata=np.stack([
            np.arange(P),
            x_coords,
            y_coords,
            np.degrees(angles) % 360,
        ], axis=-1),
        hovertemplate=(
            "Token: %{customdata[0]:.0f}<br>"
            f"Dim {dim_x}: %{{customdata[1]:.4f}}<br>"
            f"Dim {dim_y}: %{{customdata[2]:.4f}}<br>"
            "Winkel: %{customdata[3]:.1f}°<br>"
            "<extra></extra>"
        ),
        name="Token Embeddings",
    ))

    # Fit and draw a circle to show the circular structure
    radii = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    avg_radius = radii.mean()

    theta = np.linspace(0, 2*np.pi, 200)
    circle_x = cx + avg_radius * np.cos(theta)
    circle_y = cy + avg_radius * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode="lines",
        line=dict(color="rgba(255,0,0,0.3)", width=2, dash="dash"),
        name=f"Fitted circle (r={avg_radius:.3f})",
        hoverinfo="skip",
    ))

    # Draw lines connecting consecutive tokens to show winding order
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
        xaxis=dict(scaleanchor="y", scaleratio=1),
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
        """Train model with live plot updates. Does NOT auto-save."""
        P, d_model, n_heads, d_mlp, epochs = int(P), int(d_model), int(n_heads), int(d_mlp), int(epochs)
        train_frac, lr, weight_decay = float(train_frac), float(lr), float(weight_decay)

        logs = []
        train_losses = []
        test_accs = []
        metrics_table = []

        model = ModularAdditionTransformer(P=P, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Create dataset
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

                fig = make_training_plot(train_losses, test_accs)
                df = pd.DataFrame(metrics_table)
                yield fig, df, "\n".join(logs[-20:])

                if test_acc > 0.99:
                    logs.append(f"🎉 GROKKED at epoch {epoch}! (test_acc={test_acc:.4f})")
                    break

        # Store in state for manual saving later (NO auto-save)
        model.eval()
        state["model"] = model
        state["train_losses"] = train_losses
        state["test_accs"] = test_accs
        state["metrics_table"] = metrics_table
        state["train_idx"] = train_idx
        state["test_idx"] = test_idx
        state["optimizer_state"] = optimizer.state_dict()
        state["config"] = {
            "P": P, "d_model": d_model, "n_heads": n_heads, "d_mlp": d_mlp,
            "train_frac": train_frac, "epochs": epochs, "lr": lr, "weight_decay": weight_decay,
        }

        # Final yield
        grokked = is_grokked(metrics_table)
        status = "GROKKED ✅" if grokked else "NOT GROKKED ❌"
        logs.append(f"\nTraining complete. Status: {status}")
        logs.append(f"Use the 'Save Run' button to save this model.")

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
                        p_input = gr.Number(value=DEFAULT_P, label="Prime P (modulus)", precision=0)
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

            # ===== TAB: Saved Runs =====
            with gr.TabItem("💾 Saved Runs"):
                gr.Markdown("### Save & Load Training Runs")
                gr.Markdown(
                    "Runs are **not** auto-saved. After training, click **Save Current Run** to persist "
                    "the model, optimizer state, train/test split, and all metrics. "
                    "Grokked models (≥99% test acc) are auto-detected and named accordingly.\n\n"
                    "**Naming:** `grokked_mod_113_1`, `grokked_mod_113_2`, `ungrokked_mod_11_1`, etc."
                )

                gr.Markdown("---")
                gr.Markdown("#### Save Current Run")
                with gr.Row():
                    custom_name_input = gr.Textbox(
                        label="Custom name (leave blank for auto-naming)",
                        placeholder="e.g. grokked_mod_113_lowlr",
                        value="",
                    )
                    save_btn = gr.Button("💾 Save Current Run", variant="primary", size="lg")
                save_status = gr.Markdown()

                def do_save(custom_name):
                    if state["model"] is None:
                        return "⚠️ **No model in memory!** Train a model first."

                    name = custom_name.strip() if custom_name.strip() else None
                    run_id = save_run(
                        model=state["model"],
                        train_losses=state["train_losses"],
                        test_accs=state["test_accs"],
                        metrics_table=state["metrics_table"],
                        config=state["config"],
                        train_idx=state.get("train_idx"),
                        test_idx=state.get("test_idx"),
                        optimizer_state=state.get("optimizer_state"),
                        custom_name=name,
                    )
                    grokked = is_grokked(state["metrics_table"])
                    status = "GROKKED ✅" if grokked else "not grokked"
                    return f"✅ **Saved as `{run_id}`** ({status}, P={state['config']['P']})"

                save_btn.click(fn=do_save, inputs=[custom_name_input], outputs=[save_status])

                gr.Markdown("---")
                gr.Markdown("#### Load a Saved Run")
                refresh_btn = gr.Button("🔄 Refresh Run List")
                
                runs_table = gr.Dataframe(
                    label="Available Runs",
                    interactive=False,
                    headers=["run_id", "P", "grokked", "final_test_acc", "d_model", "n_heads", "d_mlp", "timestamp"],
                )
                
                run_dropdown = gr.Dropdown(choices=[], label="Select a run to load")
                with gr.Row():
                    load_run_btn = gr.Button("📂 Load Selected Run", variant="primary")
                    delete_run_btn = gr.Button("🗑️ Delete Selected Run", variant="stop")
                
                loaded_info = gr.Markdown()
                loaded_plot = gr.Plot()
                loaded_table = gr.Dataframe(label="Loaded Metrics")

                def refresh_runs():
                    runs = list_saved_runs()
                    if not runs:
                        return (
                            pd.DataFrame({"message": ["No saved runs found."]}),
                            gr.Dropdown(choices=[]),
                        )
                    df = pd.DataFrame(runs)
                    choices = [r["run_id"] for r in runs]
                    return df, gr.Dropdown(choices=choices, value=choices[0] if choices else None)

                def load_selected(run_id):
                    if not run_id:
                        return "No run selected.", go.Figure(), pd.DataFrame()

                    try:
                        data = load_run(run_id)
                    except FileNotFoundError as e:
                        return f"❌ {e}", go.Figure(), pd.DataFrame()

                    # Put into global state so all other tabs can use it
                    state["model"] = data["model"]
                    state["train_losses"] = data["train_losses"]
                    state["test_accs"] = data["test_accs"]
                    state["metrics_table"] = data["metrics_df"].to_dict("records")
                    state["config"] = data["config"]
                    state["train_idx"] = data["train_idx"]
                    state["test_idx"] = data["test_idx"]
                    state["optimizer_state"] = data["optimizer_state"]

                    fig = make_training_plot(data["train_losses"], data["test_accs"])
                    meta = data["meta"]
                    config = data["config"]

                    info_md = (
                        f"### ✅ Loaded: `{run_id}`\n\n"
                        f"| Property | Value |\n"
                        f"|----------|-------|\n"
                        f"| P (modulus) | {config['P']} |\n"
                        f"| d_model | {config['d_model']} |\n"
                        f"| n_heads | {config['n_heads']} |\n"
                        f"| d_mlp | {config['d_mlp']} |\n"
                        f"| Train fraction | {config.get('train_frac', 'N/A')} |\n"
                        f"| LR | {config.get('lr', 'N/A')} |\n"
                        f"| Weight decay | {config.get('weight_decay', 'N/A')} |\n"
                        f"| Grokked | {'✅ Yes' if meta.get('grokked') else '❌ No'} |\n"
                        f"| Final test acc | {meta.get('final_test_acc', 'N/A')} |\n"
                        f"| Epochs trained | {meta.get('epochs_trained', 'N/A')} |\n"
                        f"| Saved at | {meta.get('timestamp', 'N/A')} |\n"
                        f"| Has optimizer state | {'Yes' if data['optimizer_state'] else 'No'} |\n"
                        f"| Has train/test split | {'Yes' if data['train_idx'] is not None else 'No'} |\n\n"
                        f"**Model is now loaded into memory.** You can:\n"
                        f"- Run **Fourier Discovery** or **ACDC** on it\n"
                        f"- Use **Run Inference** to test predictions\n"
                        f"- Use **Live Activations** to inspect internals\n"
                        f"- Modify and re-save under a new name\n"
                    )

                    return info_md, fig, data["metrics_df"]

                def delete_selected(run_id):
                    if not run_id:
                        return "No run selected."
                    success = delete_run(run_id)
                    if success:
                        return f"🗑️ Deleted `{run_id}`"
                    return f"❌ Could not delete `{run_id}`"

                refresh_btn.click(fn=refresh_runs, inputs=[], outputs=[runs_table, run_dropdown])
                load_run_btn.click(fn=load_selected, inputs=[run_dropdown], outputs=[loaded_info, loaded_plot, loaded_table])
                delete_run_btn.click(fn=delete_selected, inputs=[run_dropdown], outputs=[loaded_info])

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

            # ===== TAB: Circuit Equations (Live) =====
            with gr.TabItem("🔢 Circuit Equations (Live)"):
                build_equation_viewer_tab(state)

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
                inspect_circle_btn = gr.Button("🔎 Inspect Selected Circle")

                single_circle_plot = gr.Plot(label="Detailed Circle View")

                gr.Markdown("---")
                gr.Markdown("#### 📐 Projection Explanation — What Are You Actually Looking At?")
                gr.Markdown(
                    "The plot below shows **exactly** how the circle coordinates are computed. "
                    "It reveals that each point on the circle is a **weighted sum of ALL embedding "
                    "dimensions**, not a single raw dimension. This is why raw dimension pairs "
                    "look like random scatter — the circular structure lives in a specific 2D "
                    "subspace that is NOT aligned with any coordinate axis."
                )
                projection_explanation_plot = gr.Plot(
                    label="Projection Explanation (4-panel breakdown)"
                )
                projection_equation_md = gr.Markdown(
                    value="*Select and inspect a circle above to see the full mathematical breakdown.*",
                    label="Full Equation Breakdown",
                )

                # State for discovered pairs
                discovered_pairs_state = gr.State([])

                gr.Markdown("---")
                gr.Markdown("""#### Manual Override — Raw Embedding Dimensions

                ⚠️ **These are raw individual dimensions** (e.g., `W_E[:, 5]` vs `W_E[:, 12]`).
                They will usually **NOT** look like circles because the circular structure is
                encoded across *all* dimensions simultaneously.

                The circles above are found via **Fourier projections** or **PCA** — these are
                weighted sums of all dimensions, not single coordinate axes.

                **Why?** A single raw dimension contains contributions from ALL frequencies mixed together:

                $$W_E[t, d] = \\sum_{k=0}^{P/2} \\left[ \\alpha_{k,d} \\cos\\left(\\frac{2\\pi k t}{P}\\right) + \\beta_{k,d} \\sin\\left(\\frac{2\\pi k t}{P}\\right) \\right]$$

                The Fourier projection **isolates** one frequency $k$ by projecting out all others.
                """)

                with gr.Row():
                    dim_x_select = gr.Number(value=0, label="X Dimension", precision=0)
                    dim_y_select = gr.Number(value=1, label="Y Dimension", precision=0)
                    token_select = gr.Number(value=0, label="Token ID (for all-dims view)", precision=0)

                with gr.Row():
                    embed_circle_btn = gr.Button("Show Manual Pair (raw dims)")
                    embed_alldims_btn = gr.Button("Show All Dimensions for Token")

                embed_circle_plot = gr.Plot(label="Raw Embedding Dimensions (usually NOT a circle)")
                embed_alldims_plot = gr.Plot(label="All Embedding Dimensions")


                gr.Markdown("---")
                gr.Markdown("#### 📐 Winkelanalyse")

                angle_btn = gr.Button("📐 Winkel berechnen", variant="secondary")
                angle_table_output = gr.Dataframe(label="Absolute Winkel pro Token", interactive=False)
                angle_diff_table_output = gr.Dataframe(label="Paarweise Winkeldifferenzen (i → j)", interactive=False)
                angle_visual_plot = gr.Plot(label="Winkelverteilung (visuell)")
                angle_heatmap_plot = gr.Plot(label="Winkeldifferenz-Heatmap")

                def compute_angles_for_selected(selection, pairs):
                    if state["model"] is None or not pairs or not selection:
                        empty = pd.DataFrame({"error": ["Kein Kreis ausgewählt"]})
                        return empty, empty, go.Figure(), go.Figure()
                    try:
                        idx = int(selection.split(":")[0])
                    except (ValueError, IndexError):
                        empty = pd.DataFrame({"error": ["Ungültige Auswahl"]})
                        return empty, empty, go.Figure(), go.Figure()
                    if idx >= len(pairs):
                        empty = pd.DataFrame({"error": ["Index außerhalb des Bereichs"]})
                        return empty, empty, go.Figure(), go.Figure()

                    angle_table, diff_df, angle_fig, heatmap_fig = compute_circle_angles(
                        state["model"], pairs[idx]
                    )
                    return angle_table, diff_df, angle_fig, heatmap_fig

                angle_btn.click(
                    fn=compute_angles_for_selected,
                    inputs=[circle_dropdown, discovered_pairs_state],
                    outputs=[angle_table_output, angle_diff_table_output, angle_visual_plot, angle_heatmap_plot],
                )


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
                    """
                    Inspect a selected circle: show the plot, the projection explanation plot,
                    and the full mathematical breakdown.
                    """
                    if state["model"] is None or not pairs or not selection:
                        return go.Figure(), go.Figure(), "*No circle selected.*"

                    # Parse index from selection string
                    try:
                        idx = int(selection.split(":")[0])
                    except (ValueError, IndexError):
                        return go.Figure(), go.Figure(), "*Invalid selection.*"

                    if idx >= len(pairs):
                        return go.Figure(), go.Figure(), "*Index out of range.*"

                    pair = pairs[idx]
                    
                    # 1. The circle plot itself
                    circle_fig = make_auto_circle_plot(state["model"], pair)
                    
                    # 2. The educational 4-panel explanation plot
                    explanation_fig = make_projection_explanation_plot(state["model"], pair)
                    
                    # 3. The full markdown equation breakdown
                    equation_md = make_projection_equation_markdown(state["model"], pair)
                    
                    return circle_fig, explanation_fig, equation_md

                def show_embed_circle(dim_x, dim_y):
                    """Show raw embedding dimensions with a clear warning about what this is."""
                    if state["model"] is None:
                        return go.Figure()
                    
                    model = state["model"]
                    P = model.P
                    d_model = model.d_model
                    dim_x, dim_y = int(dim_x), int(dim_y)
                    
                    fig = make_embedding_circle_plot(model, dim_x, dim_y)
                    
                    # Compute circularity score for this raw pair
                    W_E = model.embed.weight[:P].detach().cpu().numpy()
                    x_coords = W_E[:, dim_x]
                    y_coords = W_E[:, dim_y]
                    cx, cy = x_coords.mean(), y_coords.mean()
                    radii = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
                    mean_r = radii.mean()
                    circularity = 1.0 - (radii.std() / (mean_r + 1e-10)) if mean_r > 1e-10 else 0.0
                    
                    # Update title with warning
                    fig.update_layout(
                        title=(
                            f"⚠️ RAW Dimensions {dim_x} vs {dim_y} — Circularity: {circularity:.4f}<br>"
                            f"<sub>This plots W_E[t, {dim_x}] vs W_E[t, {dim_y}] — just 2 of {d_model} dims. "
                            f"The circle lives in a different 2D subspace (use Auto-Find above).</sub>"
                        ),
                    )
                    
                    # Add text annotation explaining
                    fig.add_annotation(
                        text=(
                            f"This is NOT a Fourier projection.<br>"
                            f"x = W_E[t, {dim_x}] (one scalar)<br>"
                            f"y = W_E[t, {dim_y}] (one scalar)<br>"
                            f"Circularity: {circularity:.4f}<br>"
                            f"(Auto-circles typically score >0.95)"
                        ),
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        showarrow=False,
                        font=dict(size=10, color="red"),
                        bgcolor="rgba(255,255,200,0.9)",
                        bordercolor="red",
                        borderwidth=1,
                    )
                    
                    return fig

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
                    outputs=[single_circle_plot, projection_explanation_plot, projection_equation_md],
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

# =============================================================================
# Save / Load System (replaces existing save_run, list_saved_runs, load_run)
# =============================================================================

RUNS_DIR = "saved_runs"
os.makedirs(RUNS_DIR, exist_ok=True)


def _next_run_name(P: int, grokked: bool) -> str:
    """
    Auto-generate a run name like:
      grokked_mod_113_1, grokked_mod_113_2, ...
      ungrokked_mod_113_1, ...
    """
    prefix = "grokked" if grokked else "ungrokked"
    base = f"{prefix}_mod_{P}"
    
    existing = []
    runs_path = Path(RUNS_DIR)
    if runs_path.exists():
        for d in runs_path.iterdir():
            if d.is_dir() and d.name.startswith(base + "_"):
                try:
                    num = int(d.name.split("_")[-1])
                    existing.append(num)
                except ValueError:
                    pass
    
    next_num = max(existing, default=0) + 1
    return f"{base}_{next_num}"


def is_grokked(metrics_table: list[dict], threshold: float = 0.99) -> bool:
    """Detect grokking: test accuracy >= threshold."""
    if not metrics_table:
        return False
    final_test_acc = metrics_table[-1].get("test_acc", 0.0)
    return final_test_acc >= threshold


def save_run(model, train_losses, test_accs, metrics_table, config: dict,
             train_idx=None, test_idx=None, optimizer_state=None,
             custom_name: str = None) -> str:
    """
    Save a complete training run with everything needed to reload and experiment.
    
    Saves:
      - model.pt: full model state_dict
      - config.json: all hyperparameters (P, d_model, n_heads, d_mlp, lr, wd, etc.)
      - metrics.csv: epoch-by-epoch training metrics
      - curves.json: raw loss/accuracy curves for plotting
      - split.pt: the exact train/test index split (so you can evaluate on same split)
      - optimizer.pt: optimizer state_dict (so you can resume training if desired)
      - meta.json: run metadata (name, timestamp, grokked status, etc.)
    
    Does NOT auto-save. Call this explicitly.
    
    Args:
        custom_name: Override the auto-generated name. If None, auto-names.
    
    Returns:
        run_id (str): The name/ID of the saved run.
    """
    grokked = is_grokked(metrics_table)
    P = config.get("P", DEFAULT_P)
    
    if custom_name:
        run_id = custom_name
    else:
        run_id = _next_run_name(P, grokked)
    
    run_dir = Path(RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Model weights
    torch.save(model.state_dict(), run_dir / "model.pt")
    
    # 2. Config (everything needed to reconstruct the architecture)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # 3. Metrics table
    df = pd.DataFrame(metrics_table)
    df.to_csv(run_dir / "metrics.csv", index=False)
    
    # 4. Raw curves for plotting
    with open(run_dir / "curves.json", "w") as f:
        json.dump({"train_losses": train_losses, "test_accs": test_accs}, f)
    
    # 5. Train/test split indices (critical for reproducibility)
    if train_idx is not None and test_idx is not None:
        torch.save({"train_idx": train_idx, "test_idx": test_idx}, run_dir / "split.pt")
    
    # 6. Optimizer state (for resuming training)
    if optimizer_state is not None:
        torch.save(optimizer_state, run_dir / "optimizer.pt")
    
    # 7. Metadata
    meta = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "grokked": grokked,
        "final_test_acc": metrics_table[-1]["test_acc"] if metrics_table else 0.0,
        "final_train_acc": metrics_table[-1].get("train_acc", 0.0) if metrics_table else 0.0,
        "epochs_trained": metrics_table[-1]["epoch"] if metrics_table else 0,
        "P": P,
        "d_model": config.get("d_model"),
        "n_heads": config.get("n_heads"),
        "d_mlp": config.get("d_mlp"),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    return run_id


def list_saved_runs() -> list[dict]:
    """List all saved runs with summary info, sorted by recency."""
    runs = []
    runs_path = Path(RUNS_DIR)
    if not runs_path.exists():
        return runs
    
    for run_dir in sorted(runs_path.iterdir(), reverse=True):
        if run_dir.is_dir() and (run_dir / "meta.json").exists():
            with open(run_dir / "meta.json") as f:
                meta = json.load(f)
            runs.append(meta)
        elif run_dir.is_dir() and (run_dir / "config.json").exists():
            # Backwards compat with old format
            with open(run_dir / "config.json") as f:
                config = json.load(f)
            metrics_path = run_dir / "metrics.csv"
            final_acc = 0.0
            if metrics_path.exists():
                df = pd.read_csv(metrics_path)
                if len(df) > 0:
                    final_acc = df["test_acc"].iloc[-1]
            runs.append({
                "run_id": run_dir.name,
                "timestamp": "unknown",
                "grokked": final_acc >= 0.99,
                "final_test_acc": final_acc,
                "P": config.get("P"),
                "d_model": config.get("d_model"),
                "n_heads": config.get("n_heads"),
                "d_mlp": config.get("d_mlp"),
            })
    
    return runs


def load_run(run_id: str) -> dict:
    """
    Load a saved run. Returns everything needed to experiment without retraining.
    
    Returns dict with keys:
      - model: loaded ModularAdditionTransformer (eval mode)
      - config: hyperparameter dict
      - metrics_df: pandas DataFrame of training metrics
      - train_losses: list of (epoch, loss) tuples
      - test_accs: list of (epoch, acc) tuples
      - train_idx: tensor of training indices (or None)
      - test_idx: tensor of test indices (or None)
      - optimizer_state: optimizer state_dict (or None)
      - meta: metadata dict
    """
    run_dir = Path(RUNS_DIR) / run_id
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run '{run_id}' not found in {RUNS_DIR}/")
    
    # Config
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    
    # Reconstruct model
    model = ModularAdditionTransformer(
        P=config["P"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_mlp=config["d_mlp"],
    )
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    
    # Curves
    with open(run_dir / "curves.json") as f:
        curves = json.load(f)
    
    # Metrics
    metrics_df = pd.read_csv(run_dir / "metrics.csv")
    
    # Split (optional)
    train_idx, test_idx = None, None
    split_path = run_dir / "split.pt"
    if split_path.exists():
        split_data = torch.load(split_path, map_location="cpu", weights_only=True)
        train_idx = split_data["train_idx"]
        test_idx = split_data["test_idx"]
    
    # Optimizer (optional)
    optimizer_state = None
    opt_path = run_dir / "optimizer.pt"
    if opt_path.exists():
        optimizer_state = torch.load(opt_path, map_location="cpu", weights_only=True)
    
    # Meta
    meta = {}
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    
    return {
        "model": model,
        "config": config,
        "metrics_df": metrics_df,
        "train_losses": curves["train_losses"],
        "test_accs": curves["test_accs"],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "optimizer_state": optimizer_state,
        "meta": meta,
    }


def delete_run(run_id: str) -> bool:
    """Delete a saved run."""
    run_dir = Path(RUNS_DIR) / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
        return True
    return False


def rename_run(old_id: str, new_id: str) -> bool:
    """Rename a saved run."""
    old_dir = Path(RUNS_DIR) / old_id
    new_dir = Path(RUNS_DIR) / new_id
    if old_dir.exists() and not new_dir.exists():
        old_dir.rename(new_dir)
        # Update meta.json
        meta_path = new_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            meta["run_id"] = new_id
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        return True
    return False

def make_projection_explanation_plot(model: ModularAdditionTransformer,
                                      pair_info: dict) -> go.Figure:
    """
    Create an educational plot that shows EXACTLY what a circle projection is doing.

    Shows:
    - The projection equation with real coefficient values
    - A side-by-side: raw dims vs projected coords
    - The projection vectors themselves (what directions in 128D space we're projecting onto)

    This makes it crystal clear WHY auto-circles look different from raw dimension plots.

    === THE MATH ===

    For a Fourier projection at frequency k:

        x_i = (W_E[i, :] · cos_direction) / ||cos_direction||²
        y_i = (W_E[i, :] · sin_direction) / ||sin_direction||²

    where:
        cos_direction[d] = Σ_t cos(2πkt/P) · W_E[t, d]   (the "cos_k" row of Fourier-transformed W_E)
        sin_direction[d] = Σ_t sin(2πkt/P) · W_E[t, d]   (the "sin_k" row of Fourier-transformed W_E)

    So each x_i is a WEIGHTED SUM of ALL 128 dimensions of token i's embedding,
    where the weights come from the Fourier basis projected through W_E.

    For PCA:
        x_i = U[i, pc_x] * S[pc_x]    (i-th token's score on principal component pc_x)
        y_i = U[i, pc_y] * S[pc_y]

    Each PC is itself a linear combination of all d_model dimensions:
        PC_j = V[j, :]  (a unit vector in R^d_model)
        x_i = (W_E[i, :] - mean) · V[pc_x, :]

    For raw dimensions:
        x_i = W_E[i, dim_x]    (just one number from the embedding vector)
        y_i = W_E[i, dim_y]    (just one number)
    """
    P = model.P
    d_model = model.d_model
    W_E = model.embed.weight[:P].detach().cpu().numpy()  # (P, d_model)

    method = pair_info["method"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Circle Plot ({method} projection)",
            "Projection Vector Weights (what dims contribute)",
            "Raw Dims 0 vs 1 (for comparison — NOT a circle)",
            "How x-coordinate is computed (one token example)",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    x_coords = pair_info["x_coords"]
    y_coords = pair_info["y_coords"]
    color_values = np.arange(P) / P

    # === Panel 1: The actual circle ===
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode="markers",
        marker=dict(size=6, color=color_values, colorscale=CYCLIC_BLUE_GREEN, showscale=False),
        text=[str(i) for i in range(P)],
        hovertemplate="Token %{text}: (%{x:.3f}, %{y:.3f})<extra></extra>",
        name="Projected (circle)",
    ), row=1, col=1)

    # Fitted circle overlay
    cx, cy = x_coords.mean(), y_coords.mean()
    radii = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    avg_r = radii.mean()
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=cx + avg_r * np.cos(theta), y=cy + avg_r * np.sin(theta),
        mode="lines", line=dict(color="rgba(255,0,0,0.3)", dash="dash"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)

    # === Panel 2: Projection vector weights ===
    if method == "fourier":
        # Show the actual cos_direction and sin_direction vectors
        k = pair_info.get("frequency", 1)

        # Reconstruct the Fourier basis vectors
        fourier_basis = np.zeros((P, P))
        fourier_basis[0] = np.ones(P) / np.sqrt(P)
        for freq in range(1, P // 2 + 1):
            fourier_basis[2*freq - 1] = np.cos(2 * np.pi * freq * np.arange(P) / P) * np.sqrt(2/P)
            if 2*freq < P:
                fourier_basis[2*freq] = np.sin(2 * np.pi * freq * np.arange(P) / P) * np.sqrt(2/P)

        W_E_fourier = fourier_basis @ W_E  # (P, d_model)
        cos_direction = W_E_fourier[2*k - 1]  # (d_model,)
        sin_direction = W_E_fourier[2*k] if 2*k < P else np.zeros(d_model)

        # Show top contributing dimensions
        fig.add_trace(go.Bar(
            x=list(range(d_model)), y=cos_direction,
            name=f"cos_{k} direction",
            marker_color="blue", opacity=0.7,
            hovertemplate="Dim %{x}: weight=%{y:.4f}<extra>cos direction</extra>",
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=list(range(d_model)), y=sin_direction,
            name=f"sin_{k} direction",
            marker_color="orange", opacity=0.7,
            hovertemplate="Dim %{x}: weight=%{y:.4f}<extra>sin direction</extra>",
        ), row=1, col=2)

    elif method == "pca":
        from scipy.linalg import svd
        W_centered = W_E - W_E.mean(axis=0, keepdims=True)
        U, S, Vt = svd(W_centered, full_matrices=False)

        pc_i = pair_info.get("pc_i", 0)
        pc_j = pair_info.get("pc_j", 1)

        # The PC directions in embedding space
        fig.add_trace(go.Bar(
            x=list(range(d_model)), y=Vt[pc_i],
            name=f"PC_{pc_i} direction (σ={S[pc_i]:.2f})",
            marker_color="blue", opacity=0.7,
            hovertemplate="Dim %{x}: weight=%{y:.4f}<extra>PC_x direction</extra>",
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=list(range(d_model)), y=Vt[pc_j],
            name=f"PC_{pc_j} direction (σ={S[pc_j]:.2f})",
            marker_color="orange", opacity=0.7,
            hovertemplate="Dim %{x}: weight=%{y:.4f}<extra>PC_y direction</extra>",
        ), row=1, col=2)

    elif method == "raw":
        dim_x = pair_info["dim_x"]
        dim_y = pair_info["dim_y"]
        # For raw: the "projection vector" is just a one-hot
        proj_x = np.zeros(d_model)
        proj_x[dim_x] = 1.0
        proj_y = np.zeros(d_model)
        proj_y[dim_y] = 1.0

        fig.add_trace(go.Bar(
            x=list(range(d_model)), y=proj_x,
            name=f"x-axis = dim {dim_x} (one-hot)",
            marker_color="blue",
        ), row=1, col=2)
        fig.add_trace(go.Bar(
            x=list(range(d_model)), y=proj_y,
            name=f"y-axis = dim {dim_y} (one-hot)",
            marker_color="orange",
        ), row=1, col=2)

    # === Panel 3: Raw dims 0 vs 1 for comparison (usually NOT a circle) ===
    raw_x = W_E[:, 0]
    raw_y = W_E[:, 1]
    fig.add_trace(go.Scatter(
        x=raw_x, y=raw_y,
        mode="markers",
        marker=dict(size=5, color=color_values, colorscale=CYCLIC_BLUE_GREEN, showscale=False),
        text=[str(i) for i in range(P)],
        hovertemplate="Token %{text}: (dim0=%{x:.3f}, dim1=%{y:.3f})<extra></extra>",
        name="Raw dims (0,1)",
    ), row=2, col=1)

    # === Panel 4: Show computation for ONE example token ===
    example_token = 5  # Pick token 5 as example
    token_embedding = W_E[example_token]  # (d_model,)

    if method == "fourier":
        k = pair_info.get("frequency", 1)
        # x_coord = token_embedding · cos_direction / ||cos_direction||²
        dot_product = token_embedding * cos_direction  # element-wise contribution
        norm_sq = np.linalg.norm(cos_direction)**2

        # Show per-dimension contribution to the dot product
        fig.add_trace(go.Bar(
            x=list(range(d_model)),
            y=dot_product,
            name=f"Token {example_token}: W_E[{example_token},d] × cos_dir[d]",
            marker_color=["red" if v < 0 else "green" for v in dot_product],
            hovertemplate=(
                f"Dim %{{x}}: W_E[{example_token},%{{x}}]=%{{customdata[0]:.4f}} × "
                f"cos_dir[%{{x}}]=%{{customdata[1]:.4f}} = %{{y:.4f}}<extra></extra>"
            ),
            customdata=np.stack([token_embedding, cos_direction], axis=-1),
        ), row=2, col=2)

        # Add annotation showing the final sum
        total = dot_product.sum() / (norm_sq + 1e-10)
        fig.add_annotation(
            text=(
                f"x_{example_token} = Σ(W_E[{example_token},d] × cos_dir[d]) / ||cos_dir||²<br>"
                f"= {dot_product.sum():.4f} / {norm_sq:.4f} = <b>{total:.4f}</b>"
            ),
            xref="x4", yref="y4",
            x=d_model * 0.5, y=max(abs(dot_product)) * 0.8,
            showarrow=False, font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
        )

    elif method == "pca":
        pc_i = pair_info.get("pc_i", 0)
        W_centered = W_E - W_E.mean(axis=0, keepdims=True)
        from scipy.linalg import svd
        U, S, Vt = svd(W_centered, full_matrices=False)

        # x_coord = (W_E[token] - mean) · V[pc_i] * S[pc_i]
        centered_token = W_centered[example_token]
        contributions = centered_token * Vt[pc_i]

        fig.add_trace(go.Bar(
            x=list(range(d_model)),
            y=contributions,
            name=f"Token {example_token}: (W_E[{example_token},d]-μ[d]) × PC_{pc_i}[d]",
            marker_color=["red" if v < 0 else "green" for v in contributions],
            hovertemplate=(
                f"Dim %{{x}}: centered[%{{x}}]=%{{customdata[0]:.4f}} × "
                f"PC_{pc_i}[%{{x}}]=%{{customdata[1]:.4f}} = %{{y:.4f}}<extra></extra>"
            ),
            customdata=np.stack([centered_token, Vt[pc_i]], axis=-1),
        ), row=2, col=2)

        total = contributions.sum() * S[pc_i]
        fig.add_annotation(
            text=(
                f"x_{example_token} = [Σ(centered[d] × PC_{pc_i}[d])] × σ_{pc_i}<br>"
                f"= {contributions.sum():.4f} × {S[pc_i]:.4f} = <b>{total:.4f}</b>"
            ),
            xref="x4", yref="y4",
            x=d_model * 0.5, y=max(abs(contributions)) * 0.8,
            showarrow=False, font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
        )

    elif method == "raw":
        dim_x = pair_info["dim_x"]
        # For raw: just show the single dimension value
        fig.add_trace(go.Bar(
            x=list(range(d_model)),
            y=np.where(np.arange(d_model) == dim_x, token_embedding, 0),
            name=f"Token {example_token}: only dim {dim_x} used",
            marker_color="green",
        ), row=2, col=2)

        fig.add_annotation(
            text=f"x_{example_token} = W_E[{example_token}, {dim_x}] = <b>{token_embedding[dim_x]:.4f}</b><br>(just one dimension, no projection)",
            xref="x4", yref="y4",
            x=d_model * 0.5, y=token_embedding[dim_x] * 0.8,
            showarrow=False, font=dict(size=10),
            bgcolor="rgba(255,255,255,0.8)",
        )

    # Layout
    fig.update_layout(
        height=900,
        width=1100,
        title_text=(
            f"How This Circle Is Computed — Method: {method.upper()}<br>"
            f"<sub>Left: the circle you see | Right: the projection weights across all {d_model} dims | "
            f"Bottom-left: raw dims (NOT a circle) | Bottom-right: per-dim computation for token {example_token}</sub>"
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )

    fig.update_xaxes(title_text="Dimension index", row=1, col=2)
    fig.update_xaxes(title_text=f"Raw Dim 0", row=2, col=1)
    fig.update_xaxes(title_text="Dimension index", row=2, col=2)
    fig.update_yaxes(title_text="Projection weight", row=1, col=2)

    return fig

def make_projection_equation_markdown(model: ModularAdditionTransformer,
                                       pair_info: dict) -> str:
    """
    Generate a detailed Markdown explanation of exactly what math produces
    the circle coordinates, with REAL numbers from the actual model weights.

    This is the "show your work" function that makes it impossible to be confused
    about what the auto-circle plot is showing.
    """
    P = model.P
    d_model = model.d_model
    W_E = model.embed.weight[:P].detach().cpu().numpy()
    method = pair_info["method"]

    md = f"## 🔍 Projection Explanation: `{method}` method\n\n"
    md += f"**Model:** P={P}, d_model={d_model}\n\n"
    md += "---\n\n"

    if method == "fourier":
        k = pair_info.get("frequency", 1)

        # Reconstruct the actual projection vectors
        fourier_basis = np.zeros((P, P))
        fourier_basis[0] = np.ones(P) / np.sqrt(P)
        for freq in range(1, P // 2 + 1):
            fourier_basis[2*freq - 1] = np.cos(2 * np.pi * freq * np.arange(P) / P) * np.sqrt(2/P)
            if 2*freq < P:
                fourier_basis[2*freq] = np.sin(2 * np.pi * freq * np.arange(P) / P) * np.sqrt(2/P)

        W_E_fourier = fourier_basis @ W_E
        cos_direction = W_E_fourier[2*k - 1]
        sin_direction = W_E_fourier[2*k] if 2*k < P else np.zeros(d_model)

        cos_norm_sq = np.linalg.norm(cos_direction)**2
        sin_norm_sq = np.linalg.norm(sin_direction)**2

        # Find top contributing dimensions
        top_cos_dims = np.argsort(np.abs(cos_direction))[-5:][::-1]
        top_sin_dims = np.argsort(np.abs(sin_direction))[-5:][::-1]

        md += f"### Method: Fourier Projection (frequency k={k})\n\n"
        md += "#### What the axes represent\n\n"
        md += "The x and y coordinates are **NOT** raw embedding dimensions. They are:\n\n"
        md += "$$x_t = \\frac{\\mathbf{e}_t \\cdot \\mathbf{c}_k}{\\|\\mathbf{c}_k\\|^2}$$\n\n"
        md += "$$y_t = \\frac{\\mathbf{e}_t \\cdot \\mathbf{s}_k}{\\|\\mathbf{s}_k\\|^2}$$\n\n"
        md += "where:\n"
        md += f"- $\\mathbf{{e}}_t = W_E[t, :] \\in \\mathbb{{R}}^{{{d_model}}}$ is token $t$'s full embedding vector\n"
        md += f"- $\\mathbf{{c}}_k \\in \\mathbb{{R}}^{{{d_model}}}$ is the \"cosine-{k} direction\" in embedding space\n"
        md += f"- $\\mathbf{{s}}_k \\in \\mathbb{{R}}^{{{d_model}}}$ is the \"sine-{k} direction\" in embedding space\n\n"

        md += "#### How the direction vectors are computed\n\n"
        md += "$$\\mathbf{c}_k[d] = \\sum_{t=0}^{P-1} \\cos\\!\\left(\\frac{2\\pi k t}{P}\\right) \\cdot W_E[t, d] \\cdot \\sqrt{\\frac{2}{P}}$$\n\n"
        md += "In words: for each embedding dimension $d$, we compute how much that dimension\n"
        md += f"correlates with $\\cos(2\\pi \\cdot {k} \\cdot t / {P})$ across all {P} tokens.\n\n"

        md += "#### Real values from this model\n\n"
        md += f"- $\\|\\mathbf{{c}}_{{{k}}}\\|^2 = {cos_norm_sq:.4f}$\n"
        md += f"- $\\|\\mathbf{{s}}_{{{k}}}\\|^2 = {sin_norm_sq:.4f}$\n\n"

        md += f"**Top 5 dimensions contributing to $\\mathbf{{c}}_{{{k}}}$:**\n\n"
        md += "| Dim | Weight | |Weight| |\n|-----|--------|--------|\n"
        for d in top_cos_dims:
            md += f"| {d} | {cos_direction[d]:.4f} | {abs(cos_direction[d]):.4f} |\n"

        md += f"\n**Top 5 dimensions contributing to $\\mathbf{{s}}_{{{k}}}$:**\n\n"
        md += "| Dim | Weight | |Weight| |\n|-----|--------|--------|\n"
        for d in top_sin_dims:
            md += f"| {d} | {sin_direction[d]:.4f} | {abs(sin_direction[d]):.4f} |\n"

        md += "\n#### Worked example: Token t=5\n\n"
        e5 = W_E[5]
        x5 = np.dot(e5, cos_direction) / (cos_norm_sq + 1e-10)
        y5 = np.dot(e5, sin_direction) / (sin_norm_sq + 1e-10)

        md += f"$$x_5 = \\frac{{\\mathbf{{e}}_5 \\cdot \\mathbf{{c}}_{{{k}}}}}{{\\|\\mathbf{{c}}_{{{k}}}\\|^2}} = \\frac{{{np.dot(e5, cos_direction):.4f}}}{{{cos_norm_sq:.4f}}} = {x5:.4f}$$\n\n"
        md += f"$$y_5 = \\frac{{\\mathbf{{e}}_5 \\cdot \\mathbf{{s}}_{{{k}}}}}{{\\|\\mathbf{{s}}_{{{k}}}\\|^2}} = \\frac{{{np.dot(e5, sin_direction):.4f}}}{{{sin_norm_sq:.4f}}} = {y5:.4f}$$\n\n"

        # Show why it's a circle
        expected_angle = 2 * np.pi * k * 5 / P
        md += f"#### Why it's a circle\n\n"
        md += f"If the model perfectly learned the Fourier representation, then:\n\n"
        md += f"$$x_t \\approx A \\cos\\!\\left(\\frac{{2\\pi \\cdot {k} \\cdot t}}{{{P}}}\\right), \\quad y_t \\approx A \\sin\\!\\left(\\frac{{2\\pi \\cdot {k} \\cdot t}}{{{P}}}\\right)$$\n\n"
        md += f"For token 5: expected angle = $2\\pi \\cdot {k} \\cdot 5 / {P} = {expected_angle:.4f}$ rad = {np.degrees(expected_angle):.1f}°\n\n"
        md += f"Actual angle from data: {np.degrees(np.arctan2(y5, x5)):.1f}°\n\n"

        md += "#### Why raw dimensions DON'T show circles\n\n"
        md += f"A single raw dimension (e.g., `W_E[t, 0]`) contains contributions from **all** frequencies mixed together:\n\n"
        md += f"$$W_E[t, 0] = \\sum_{{k'=0}}^{{{P//2}}} \\left[ \\alpha_{{k',0}} \\cos(\\omega_{{k'}} t) + \\beta_{{k',0}} \\sin(\\omega_{{k'}} t) \\right]$$\n\n"
        md += f"This superposition of many frequencies destroys the clean circular pattern.\n"
        md += f"The Fourier projection **isolates** frequency {k} by projecting out all others.\n\n"

    elif method == "pca":
        from scipy.linalg import svd
        W_centered = W_E - W_E.mean(axis=0, keepdims=True)
        U, S, Vt = svd(W_centered, full_matrices=False)

        pc_i = pair_info.get("pc_i", 0)
        pc_j = pair_info.get("pc_j", 1)

        md += f"### Method: PCA (Principal Components {pc_i} and {pc_j})\n\n"
        md += "#### What the axes represent\n\n"
        md += f"$$x_t = (\\mathbf{{e}}_t - \\bar{{\\mathbf{{e}}}}) \\cdot \\mathbf{{v}}_{{{pc_i}}} \\times \\sigma_{{{pc_i}}}$$\n\n"
        md += f"$$y_t = (\\mathbf{{e}}_t - \\bar{{\\mathbf{{e}}}}) \\cdot \\mathbf{{v}}_{{{pc_j}}} \\times \\sigma_{{{pc_j}}}$$\n\n"
        md += "where:\n"
        md += f"- $\\bar{{\\mathbf{{e}}}} \\in \\mathbb{{R}}^{{{d_model}}}$ is the mean embedding across all {P} tokens\n"
        md += f"- $\\mathbf{{v}}_{{{pc_i}}} \\in \\mathbb{{R}}^{{{d_model}}}$ is the {pc_i}-th right singular vector of centered $W_E$\n"
        md += f"- $\\sigma_{{{pc_i}}} = {S[pc_i]:.4f}$ is the {pc_i}-th singular value\n"
        md += f"- $\\sigma_{{{pc_j}}} = {S[pc_j]:.4f}$ is the {pc_j}-th singular value\n\n"

        md += f"Each PC is a **linear combination of all {d_model} dimensions**.\n\n"

        # Top contributing dims for each PC
        top_dims_i = np.argsort(np.abs(Vt[pc_i]))[-5:][::-1]
        top_dims_j = np.argsort(np.abs(Vt[pc_j]))[-5:][::-1]

        md += f"**Top 5 dims in PC_{pc_i}:**\n\n"
        md += "| Dim | Weight |\n|-----|--------|\n"
        for d in top_dims_i:
            md += f"| {d} | {Vt[pc_i][d]:.4f} |\n"

        md += f"\n**Top 5 dims in PC_{pc_j}:**\n\n"
        md += "| Dim | Weight |\n|-----|--------|\n"
        for d in top_dims_j:
            md += f"| {d} | {Vt[pc_j][d]:.4f} |\n"

        md += "\n#### Worked example: Token t=5\n\n"
        centered_5 = W_centered[5]
        x5 = np.dot(centered_5, Vt[pc_i]) * S[pc_i]
        y5 = np.dot(centered_5, Vt[pc_j]) * S[pc_j]
        md += f"$$x_5 = (\\mathbf{{e}}_5 - \\bar{{\\mathbf{{e}}}}) \\cdot \\mathbf{{v}}_{{{pc_i}}} \\times {S[pc_i]:.4f} = {np.dot(centered_5, Vt[pc_i]):.4f} \\times {S[pc_i]:.4f} = {x5:.4f}$$\n\n"
        md += f"$$y_5 = (\\mathbf{{e}}_5 - \\bar{{\\mathbf{{e}}}}) \\cdot \\mathbf{{v}}_{{{pc_j}}} \\times {S[pc_j]:.4f} = {np.dot(centered_5, Vt[pc_j]):.4f} \\times {S[pc_j]:.4f} = {y5:.4f}$$\n\n"

        md += "#### Why PCA finds circles\n\n"
        md += "If the embedding encodes tokens on a circle at frequency $k$, then the two\n"
        md += "largest-variance directions will align with $\\cos(\\omega_k t)$ and $\\sin(\\omega_k t)$.\n"
        md += "PCA finds these as the top principal components because they explain the most variance.\n\n"

    elif method == "raw":
        dim_x = pair_info["dim_x"]
        dim_y = pair_info["dim_y"]

        md += f"### Method: Raw Dimension Pair ({dim_x}, {dim_y})\n\n"
        md += "#### What the axes represent\n\n"
        md += f"$$x_t = W_E[t, {dim_x}] \\quad \\text{{(just one scalar from the embedding)}}$$\n\n"
        md += f"$$y_t = W_E[t, {dim_y}] \\quad \\text{{(just one scalar from the embedding)}}$$\n\n"
        md += "This is the **simplest** case — no projection, no weighted sum.\n"
        md += f"You're literally plotting column {dim_x} vs column {dim_y} of the {P}×{d_model} embedding matrix.\n\n"
        md += "#### Why this is rare\n\n"
        md += f"Out of $\\binom{{{d_model}}}{{2}} = {d_model*(d_model-1)//2}$ possible dimension pairs, "
        md += "very few will show circular structure because the circle lives in an **arbitrary** "
        md += "2D subspace that is generally not axis-aligned.\n\n"

        md += f"#### Real values: Token 5\n\n"
        md += f"$$x_5 = W_E[5, {dim_x}] = {W_E[5, dim_x]:.6f}$$\n\n"
        md += f"$$y_5 = W_E[5, {dim_y}] = {W_E[5, dim_y]:.6f}$$\n\n"
        
        md += "#### Why this pair shows a circle (if it does)\n\n"
        md += f"This is one of the rare cases where the circular structure happens to be\n"
        md += f"partially aligned with these two coordinate axes. The circularity score\n"
        md += f"is {pair_info['circularity_score']:.4f} (1.0 = perfect circle).\n\n"
        md += f"Most of the {d_model*(d_model-1)//2} possible raw pairs will NOT show circles.\n"
    
    # === Common footer: comparison table ===
    md += "\n---\n\n"
    md += "## Summary: Why Auto-Circles ≠ Raw Dimensions\n\n"
    md += "| | Auto-Circle (Fourier/PCA) | Raw Dimension Pair |\n"
    md += "|---|---|---|\n"
    md += f"| **x-coordinate** | Weighted sum of ALL {d_model} dims | Just 1 dim (e.g., `W_E[t, 5]`) |\n"
    md += f"| **# dims used** | {d_model} | 1 |\n"
    md += "| **Isolates frequency?** | Yes (by design) | No (all freqs mixed) |\n"
    md += "| **Looks like circle?** | Almost always | Almost never |\n"
    md += "| **Math** | $x_t = \\mathbf{e}_t \\cdot \\mathbf{v}$ (dot product) | $x_t = W_E[t, d]$ (single entry) |\n\n"
    
    md += "### Analogy\n\n"
    md += "Imagine a 3D helix (spiral staircase). Looking at it from above (a specific 2D projection), "
    md += "you see a perfect circle. But if you just plot the x-coordinate vs the z-coordinate "
    md += "(two raw axes), you see a sine wave, not a circle. The Fourier/PCA projections are like "
    md += "choosing the right viewing angle to see the circle.\n"
    
    return md

def make_raw_vs_projected_comparison(model: ModularAdditionTransformer,
                                      pair_info: dict,
                                      raw_dim_x: int = 0,
                                      raw_dim_y: int = 1) -> go.Figure:
    """
    Side-by-side comparison: the SAME tokens plotted in
    (1) the auto-discovered projection (circle), and
    (2) raw embedding dimensions (usually random scatter).
    
    This makes it viscerally obvious that the difference is purely
    about WHICH 2D subspace you're looking at.
    
    === Key Insight ===
    
    Both plots show the SAME P points (one per token).
    Both plots use the SAME underlying data (the P × d_model embedding matrix).
    The ONLY difference is the 2D subspace we project onto:
    
    Auto-circle:  x_t = W_E[t, :] · v_x    (dot product with learned direction)
                  y_t = W_E[t, :] · v_y    (dot product with another direction)
    
    Raw dims:     x_t = W_E[t, dim_x]      (just one entry)
                  y_t = W_E[t, dim_y]      (just one entry)
    
    The first is like looking at a helix from above (you see a circle).
    The second is like looking at it from a random angle (you see noise).
    """
    P = model.P
    d_model = model.d_model
    W_E = model.embed.weight[:P].detach().cpu().numpy()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Auto-Discovered Projection ({pair_info['method']})",
            f"Raw Dimensions ({raw_dim_x} vs {raw_dim_y})",
        ),
        horizontal_spacing=0.1,
    )
    
    color_values = np.arange(P) / P
    
    # Left: the circle (projected)
    x_proj = pair_info["x_coords"]
    y_proj = pair_info["y_coords"]
    
    fig.add_trace(go.Scatter(
        x=x_proj, y=y_proj,
        mode="markers",
        marker=dict(size=6, color=color_values, colorscale=CYCLIC_BLUE_GREEN, showscale=False),
        text=[str(i) for i in range(P)],
        hovertemplate="Token %{text}: (%{x:.3f}, %{y:.3f})<extra>PROJECTED</extra>",
        name="Projected (circle)",
    ), row=1, col=1)
    
    # Fitted circle on left
    cx, cy = x_proj.mean(), y_proj.mean()
    radii = np.sqrt((x_proj - cx)**2 + (y_proj - cy)**2)
    avg_r = radii.mean()
    theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=cx + avg_r * np.cos(theta), y=cy + avg_r * np.sin(theta),
        mode="lines", line=dict(color="rgba(255,0,0,0.3)", dash="dash"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    
    # Right: raw dimensions (usually NOT a circle)
    x_raw = W_E[:, raw_dim_x]
    y_raw = W_E[:, raw_dim_y]
    
    fig.add_trace(go.Scatter(
        x=x_raw, y=y_raw,
        mode="markers",
        marker=dict(size=6, color=color_values, colorscale=CYCLIC_BLUE_GREEN, showscale=False),
        text=[str(i) for i in range(P)],
        hovertemplate="Token %{text}: (%{x:.3f}, %{y:.3f})<extra>RAW DIMS</extra>",
        name=f"Raw dims ({raw_dim_x}, {raw_dim_y})",
    ), row=1, col=2)
    
    # Attempt fitted circle on right (to show it doesn't fit)
    cx_r, cy_r = x_raw.mean(), y_raw.mean()
    radii_r = np.sqrt((x_raw - cx_r)**2 + (y_raw - cy_r)**2)
    avg_r_r = radii_r.mean()
    fig.add_trace(go.Scatter(
        x=cx_r + avg_r_r * np.cos(theta), y=cy_r + avg_r_r * np.sin(theta),
        mode="lines", line=dict(color="rgba(255,0,0,0.3)", dash="dash"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=2)
    
    # Compute circularity scores for annotation
    circ_proj = pair_info["circularity_score"]
    circ_raw = 1.0 - (radii_r.std() / (radii_r.mean() + 1e-10))
    
    fig.update_layout(
        height=500,
        width=1000,
        title_text=(
            f"Same {P} tokens, same embedding matrix — different 2D subspaces<br>"
            f"<sub>Left circularity: {circ_proj:.4f} | Right circularity: {circ_raw:.4f} | "
            f"Both use W_E ∈ ℝ^({P}×{d_model})</sub>"
        ),
    )
    
    # Add annotations explaining the math
    fig.add_annotation(
        text=f"x = W_E[t,:] · v_cos<br>y = W_E[t,:] · v_sin<br>(sum of {d_model} dims)",
        xref="x1", yref="y1",
        x=cx, y=cy,
        showarrow=False, font=dict(size=9),
        bgcolor="rgba(255,255,255,0.8)",
    )
    
    fig.add_annotation(
        text=f"x = W_E[t, {raw_dim_x}]<br>y = W_E[t, {raw_dim_y}]<br>(just 1 dim each)",
        xref="x2", yref="y2",
        x=cx_r, y=cy_r,
        showarrow=False, font=dict(size=9),
        bgcolor="rgba(255,255,255,0.8)",
    )
    
    return fig

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
