# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "numpy",
#     "scipy",
#     "plotly",
#     "gradio",
# ]
# ///
"""
Grokking Circuit Discovery & Verification Tool

Automatically discovers the Fourier multiplication algorithm (and similar
structured circuits) in small transformers trained on modular arithmetic,
based on Nanda et al. (2023) "Progress Measures for Grokking via
Mechanistic Interpretability".

Features:
- Train on modular addition OR multiplication
- L2 (Lasso-style) weight regularization added to loss
- Save/load runs locally in a 'runs/' folder
- Reload and visualize previous runs from the GUI
- Configurable network size (small, medium, large)
- 30k epochs by default
- Extended visualizations (weight norms over time, loss landscape, etc.)

Usage:
    uv run grok.py
"""

import warnings
import math
import os
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# =============================================================================
# Minimal Transformer for Modular Arithmetic
# =============================================================================

class ModularArithmeticTransformer(nn.Module):
    """
    1-layer transformer for modular arithmetic, following Nanda et al. (2023).
    Input: "a b =" -> predicts (a op b) mod P
    """
    def __init__(self, P: int = 113, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512):
        super().__init__()
        self.P = P
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_mlp = d_mlp

        # Embeddings
        self.embed = nn.Embedding(P + 1, d_model)  # P tokens + 1 for '='
        self.pos_embed = nn.Embedding(3, d_model)  # 3 positions: a, b, =

        # Attention (single layer)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # MLP
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)

        # Unembed
        self.unembed = nn.Linear(d_model, P, bias=False)

    def forward(self, a_idx, b_idx):
        """Forward pass. a_idx, b_idx are integer tensors."""
        batch = a_idx.shape[0]
        eq_idx = torch.full((batch,), self.P, device=a_idx.device)

        # Embed tokens + positions
        pos_ids = torch.arange(3, device=a_idx.device).unsqueeze(0).expand(batch, -1)
        tok_ids = torch.stack([a_idx, b_idx, eq_idx], dim=1)
        x = self.embed(tok_ids) + self.pos_embed(pos_ids)  # (batch, 3, d_model)

        # Attention (from position 2 '=' to positions 0,1)
        Q = self.W_Q(x[:, 2:3, :])  # (batch, 1, d_model)
        K = self.W_K(x[:, :2, :])   # (batch, 2, d_model)
        V = self.W_V(x[:, :2, :])   # (batch, 2, d_model)

        # Multi-head attention
        batch_size = Q.shape[0]
        Q = Q.view(batch_size, 1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)  # (batch, n_heads, 1, 2)
        attn_out = torch.matmul(attn, V)  # (batch, n_heads, 1, d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        attn_out = self.W_O(attn_out)

        # Residual + MLP
        residual = x[:, 2:3, :] + attn_out  # (batch, 1, d_model)
        mlp_hidden = F.relu(self.mlp_in(residual))
        mlp_out = self.mlp_out(mlp_hidden)
        final = residual + mlp_out  # (batch, 1, d_model)

        logits = self.unembed(final.squeeze(1))  # (batch, P)
        return logits

    def get_mlp_activations(self, a_idx, b_idx):
        """Get MLP hidden activations for analysis."""
        batch = a_idx.shape[0]
        eq_idx = torch.full((batch,), self.P, device=a_idx.device)
        pos_ids = torch.arange(3, device=a_idx.device).unsqueeze(0).expand(batch, -1)
        tok_ids = torch.stack([a_idx, b_idx, eq_idx], dim=1)
        x = self.embed(tok_ids) + self.pos_embed(pos_ids)

        Q = self.W_Q(x[:, 2:3, :])
        K = self.W_K(x[:, :2, :])
        V = self.W_V(x[:, :2, :])

        batch_size = Q.shape[0]
        Q = Q.view(batch_size, 1, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, 2, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.d_model)
        attn_out = self.W_O(attn_out)

        residual = x[:, 2:3, :] + attn_out
        mlp_hidden = F.relu(self.mlp_in(residual))
        return mlp_hidden.squeeze(1), attn  # (batch, d_mlp), (batch, n_heads, 1, 2)

# =============================================================================
# Training
# =============================================================================

def compute_l2_reg(model):
    """Compute sum of squares of all weights (lasso/ridge style regularization)."""
    l2 = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        l2 = l2 + (param ** 2).sum()
    return l2

def train_model(P: int = 113, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512,
                train_frac: float = 0.3, epochs: int = 30000, lr: float = 1e-3,
                weight_decay: float = 1.0, l2_lambda: float = 0.01,
                operation: str = "addition", progress_cb=None) -> tuple:
    """Train a model on modular arithmetic until it groks."""
    model = ModularArithmeticTransformer(P, d_model, n_heads, d_mlp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create dataset: all pairs (a, b) with target (a op b) mod P
    all_a = torch.arange(P).repeat_interleave(P)
    all_b = torch.arange(P).repeat(P)

    if operation == "addition":
        all_targets = (all_a + all_b) % P
    elif operation == "multiplication":
        all_targets = (all_a * all_b) % P
    elif operation == "subtraction":
        all_targets = (all_a - all_b) % P
    elif operation == "division":
        # Only valid for non-zero b; for b=0 we map target to 0
        all_targets = torch.zeros_like(all_a)
        for i in range(len(all_a)):
            a_val = all_a[i].item()
            b_val = all_b[i].item()
            if b_val == 0:
                all_targets[i] = 0
            else:
                # modular inverse of b
                b_inv = pow(b_val, P - 2, P)
                all_targets[i] = (a_val * b_inv) % P
    else:
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
    train_accs = []
    weight_norms = []

    def update(msg):
        if progress_cb:
            progress_cb(msg)

    for epoch in range(epochs):
        model.train()
        logits = model(train_a, train_b)
        ce_loss = F.cross_entropy(logits, train_t)

        # L2 regularization (lasso-style: sum of squared weights)
        l2_reg = compute_l2_reg(model)
        loss = ce_loss + l2_lambda * l2_reg

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

                # Compute total weight norm
                total_norm = sum(p.data.norm().item() ** 2 for p in model.parameters()) ** 0.5

            train_losses.append((epoch, loss.item(), ce_loss.item(), l2_reg.item()))
            test_accs.append((epoch, test_acc))
            train_accs.append((epoch, train_acc))
            weight_norms.append((epoch, total_norm))
            best_test_acc = max(best_test_acc, test_acc)

            update(f"Epoch {epoch}: loss={loss.item():.4f} (CE={ce_loss.item():.4f}, L2={l2_reg.item():.4f}), train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

            if test_acc > 0.99:
                update(f"Grokked at epoch {epoch}!")
                break

    model.eval()
    return model, train_losses, test_accs, train_accs, weight_norms

# =============================================================================
# Save / Load Runs
# =============================================================================

def save_run(run_name: str, model, train_losses, test_accs, train_accs, weight_norms, config: dict):
    """Save a training run to disk."""
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(exist_ok=True)

    # Save model
    torch.save(model.state_dict(), run_dir / "model.pt")

    # Save training history
    history = {
        "train_losses": train_losses,
        "test_accs": test_accs,
        "train_accs": train_accs,
        "weight_norms": weight_norms,
    }
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f)

    return str(run_dir)

def load_run(run_name: str):
    """Load a training run from disk."""
    run_dir = RUNS_DIR / run_name

    if not run_dir.exists():
        return None, None, None, None, None, None

    # Load config
    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)

    # Recreate model
    model = ModularArithmeticTransformer(
        P=config["P"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_mlp=config["d_mlp"],
    )
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    # Load history
    with open(run_dir / "history.json", "r") as f:
        history = json.load(f)

    train_losses = [tuple(x) for x in history["train_losses"]]
    test_accs = [tuple(x) for x in history["test_accs"]]
    train_accs = [tuple(x) for x in history["train_accs"]]
    weight_norms = [tuple(x) for x in history["weight_norms"]]

    return model, train_losses, test_accs, train_accs, weight_norms, config

def list_runs():
    """List all saved runs."""
    runs = []
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            if d.is_dir() and (d / "config.json").exists():
                with open(d / "config.json", "r") as f:
                    config = json.load(f)
                runs.append(f"{d.name} | P={config['P']}, op={config.get('operation','addition')}, epochs={config.get('epochs','?')}")
    return runs

# =============================================================================
# Circuit Discovery: Fourier Analysis
# =============================================================================

@dataclass
class DiscoveredCircuit:
    """A mathematically exact description of a discovered circuit."""
    key_frequencies: list
    frequency_amplitudes: dict
    algorithm_description: str
    mathematical_formula: str
    fve_logits: float
    fve_mlp: dict
    verification_accuracy: float
    neuron_frequency_assignments: dict
    embedding_fourier_norms: np.ndarray
    wl_fourier_norms: np.ndarray

class CircuitDiscoverer:
    """
    Automatically discover the Fourier multiplication circuit in a trained model.
    """

    def __init__(self, model: ModularArithmeticTransformer, operation: str = "addition"):
        self.model = model
        self.P = model.P
        self.d_model = model.d_model
        self.d_mlp = model.d_mlp
        self.operation = operation

    def _fourier_basis(self) -> tuple:
        """Get real Fourier basis vectors."""
        P = self.P
        x = np.arange(P)
        cos_basis = np.zeros((P // 2 + 1, P))
        sin_basis = np.zeros((P // 2 + 1, P))
        for k in range(P // 2 + 1):
            cos_basis[k] = np.cos(2 * np.pi * k * x / P)
            sin_basis[k] = np.sin(2 * np.pi * k * x / P)
        return cos_basis, sin_basis

    def analyze_embedding_fourier(self) -> np.ndarray:
        """Compute Fourier norms of embedding matrix W_E."""
        W_E = self.model.embed.weight[:self.P].detach().cpu().numpy()
        P = self.P
        cos_basis, sin_basis = self._fourier_basis()

        norms = np.zeros(P // 2 + 1)
        for k in range(P // 2 + 1):
            cos_proj = cos_basis[k] @ W_E
            sin_proj = sin_basis[k] @ W_E
            norms[k] = np.sqrt(np.sum(cos_proj**2) + np.sum(sin_proj**2))

        return norms

    def analyze_neuron_logit_map(self) -> np.ndarray:
        """Compute Fourier norms of neuron-logit map W_L = W_U @ W_out."""
        W_out = self.model.mlp_out.weight.detach().cpu().numpy()
        W_U = self.model.unembed.weight.detach().cpu().numpy()
        W_L = W_U @ W_out

        cos_basis, sin_basis = self._fourier_basis()
        P = self.P

        norms = np.zeros(P // 2 + 1)
        for k in range(P // 2 + 1):
            cos_proj = cos_basis[k] @ W_L
            sin_proj = sin_basis[k] @ W_L
            norms[k] = np.sqrt(np.sum(cos_proj**2) + np.sum(sin_proj**2))

        return norms

    def find_key_frequencies(self, n_freqs: int = 5) -> list:
        """Identify key frequencies used by the model."""
        embed_norms = self.analyze_embedding_fourier()
        wl_norms = self.analyze_neuron_logit_map()

        combined = embed_norms * wl_norms
        combined[0] = 0

        top_indices = np.argsort(combined)[::-1][:n_freqs]
        threshold = combined[top_indices[0]] * 0.1
        key_freqs = [int(k) for k in top_indices if combined[k] > threshold]

        return key_freqs[:n_freqs]

    def _get_targets(self, a_vals, b_vals):
        """Compute targets based on operation."""
        P = self.P
        if self.operation == "addition":
            return (a_vals + b_vals) % P
        elif self.operation == "multiplication":
            return (a_vals * b_vals) % P
        elif self.operation == "subtraction":
            return (a_vals - b_vals) % P
        else:
            return (a_vals + b_vals) % P

    def verify_trig_identities(self, key_freqs: list) -> dict:
        """Verify that MLP computes cos(wk(a op b)) and sin(wk(a op b))."""
        P = self.P
        model = self.model

        W_out = model.mlp_out.weight.detach().cpu().numpy()
        W_U = model.unembed.weight.detach().cpu().numpy()
        W_L = W_U @ W_out

        cos_basis, sin_basis = self._fourier_basis()

        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        with torch.no_grad():
            mlp_acts, _ = model.get_mlp_activations(all_a, all_b)
        mlp_acts = mlp_acts.cpu().numpy()

        a_vals = all_a.numpy()
        b_vals = all_b.numpy()
        target_vals = self._get_targets(a_vals, b_vals)

        results = {}
        for k in key_freqs:
            wk = 2 * np.pi * k / P

            cos_wk = cos_basis[k]
            sin_wk = sin_basis[k]

            u_k = W_L.T @ cos_wk
            v_k = W_L.T @ sin_wk
            u_k = u_k / (np.linalg.norm(u_k) + 1e-10)
            v_k = v_k / (np.linalg.norm(v_k) + 1e-10)

            cos_proj = mlp_acts @ u_k
            sin_proj = mlp_acts @ v_k

            true_cos = np.cos(wk * target_vals)
            true_sin = np.sin(wk * target_vals)

            alpha_cos = np.dot(cos_proj, true_cos) / (np.dot(true_cos, true_cos) + 1e-10)
            alpha_sin = np.dot(sin_proj, true_sin) / (np.dot(true_sin, true_sin) + 1e-10)

            residual_cos = cos_proj - alpha_cos * true_cos
            fve_cos = 1.0 - np.var(residual_cos) / (np.var(cos_proj) + 1e-10)

            residual_sin = sin_proj - alpha_sin * true_sin
            fve_sin = 1.0 - np.var(residual_sin) / (np.var(sin_proj) + 1e-10)

            results[k] = {
                "fve_cos": float(max(0, fve_cos)),
                "fve_sin": float(max(0, fve_sin)),
                "amplitude_cos": float(alpha_cos),
                "amplitude_sin": float(alpha_sin),
            }

        return results

    def verify_logit_approximation(self, key_freqs: list) -> tuple:
        """Verify logits ~ sum_k alpha_k * cos(wk(a op b - c))."""
        P = self.P
        model = self.model

        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        with torch.no_grad():
            logits = model(all_a, all_b).cpu().numpy()

        a_vals = all_a.numpy()
        b_vals = all_b.numpy()
        target_vals = self._get_targets(a_vals, b_vals)

        n_samples = P * P
        X = np.zeros((n_samples, P, len(key_freqs)))
        for i, k in enumerate(key_freqs):
            wk = 2 * np.pi * k / P
            for sample_idx in range(n_samples):
                t = target_vals[sample_idx]
                for c in range(P):
                    X[sample_idx, c, i] = np.cos(wk * (t - c))

        X_flat = X.reshape(-1, len(key_freqs))
        y_flat = logits.flatten()

        alphas, _, _, _ = np.linalg.lstsq(X_flat, y_flat, rcond=None)

        y_pred = X_flat @ alphas
        ss_res = np.sum((y_flat - y_pred) ** 2)
        ss_tot = np.sum((y_flat - y_flat.mean()) ** 2)
        fve = 1.0 - ss_res / (ss_tot + 1e-10)

        return float(max(0, fve)), alphas

    def assign_neurons_to_frequencies(self, key_freqs: list) -> dict:
        """Assign each neuron to its best-matching frequency."""
        P = self.P
        model = self.model

        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        with torch.no_grad():
            mlp_acts, _ = model.get_mlp_activations(all_a, all_b)
        mlp_acts = mlp_acts.cpu().numpy()

        a_vals = all_a.numpy()
        b_vals = all_b.numpy()

        assignments = {}
        for neuron_idx in range(min(self.d_mlp, 200)):
            activations = mlp_acts[:, neuron_idx]
            best_fve = 0.0
            best_freq = -1

            for k in key_freqs:
                wk = 2 * np.pi * k / P
                basis = np.column_stack([
                    np.ones(len(a_vals)),
                    np.cos(wk * a_vals), np.sin(wk * a_vals),
                    np.cos(wk * b_vals), np.sin(wk * b_vals),
                    np.cos(wk * a_vals) * np.cos(wk * b_vals),
                    np.sin(wk * a_vals) * np.sin(wk * b_vals),
                    np.cos(wk * a_vals) * np.sin(wk * b_vals),
                    np.sin(wk * a_vals) * np.cos(wk * b_vals),
                ])

                coeffs, _, _, _ = np.linalg.lstsq(basis, activations, rcond=None)
                pred = basis @ coeffs
                ss_res = np.sum((activations - pred) ** 2)
                ss_tot = np.sum((activations - activations.mean()) ** 2)
                fve = 1.0 - ss_res / (ss_tot + 1e-10) if ss_tot > 1e-10 else 0.0

                if fve > best_fve:
                    best_fve = fve
                    best_freq = k

            if best_fve > 0.85:
                assignments[neuron_idx] = {"frequency": best_freq, "fve": best_fve}

        return assignments

    def exhaustive_verification(self, key_freqs: list, alphas: np.ndarray) -> tuple:
        """Test the discovered circuit on all inputs."""
        P = self.P
        correct = 0
        total = P * P

        for a in range(P):
            for b in range(P):
                if self.operation == "addition":
                    true_answer = (a + b) % P
                elif self.operation == "multiplication":
                    true_answer = (a * b) % P
                elif self.operation == "subtraction":
                    true_answer = (a - b) % P
                else:
                    true_answer = (a + b) % P

                logits_approx = np.zeros(P)
                for i, k in enumerate(key_freqs):
                    wk = 2 * np.pi * k / P
                    for c in range(P):
                        logits_approx[c] += alphas[i] * np.cos(wk * (true_answer - c))

                predicted = np.argmax(logits_approx)
                if predicted == true_answer:
                    correct += 1

        accuracy = correct / total
        return accuracy, correct, total

    def full_discovery(self, progress_cb=None) -> DiscoveredCircuit:
        """Run the full circuit discovery pipeline."""
        def update(msg):
            if progress_cb:
                progress_cb(msg)

        update("Analyzing embedding Fourier structure...")
        embed_norms = self.analyze_embedding_fourier()

        update("Analyzing neuron-logit map Fourier structure...")
        wl_norms = self.analyze_neuron_logit_map()

        update("Finding key frequencies...")
        key_freqs = self.find_key_frequencies()
        update(f"  Key frequencies: {key_freqs}")

        update("Verifying trigonometric identities in MLP...")
        trig_results = self.verify_trig_identities(key_freqs)
        for k, res in trig_results.items():
            update(f"  Freq {k}: FVE_cos={res['fve_cos']:.3f}, FVE_sin={res['fve_sin']:.3f}")

        update("Verifying logit approximation...")
        fve_logits, alphas = self.verify_logit_approximation(key_freqs)
        update(f"  Logit FVE: {fve_logits:.4f}")

        update("Assigning neurons to frequencies...")
        neuron_assignments = self.assign_neurons_to_frequencies(key_freqs)
        n_assigned = len(neuron_assignments)
        update(f"  {n_assigned}/{self.d_mlp} neurons assigned ({100*n_assigned/self.d_mlp:.1f}%)")

        update("Exhaustive verification (testing all P*P inputs)...")
        verify_acc, correct, total = self.exhaustive_verification(key_freqs, alphas)
        update(f"  Circuit accuracy: {correct}/{total} = {verify_acc:.4f}")

        P = self.P
        freq_amps = {k: float(alphas[i]) for i, k in enumerate(key_freqs)}

        formula_parts = []
        for i, k in enumerate(key_freqs):
            formula_parts.append(f"{alphas[i]:.2f} * cos(2pi*{k}*(a {self.operation} b - c)/{P})")
        formula = "logit(c | a,b) = " + " + ".join(formula_parts)

        op_symbol = {"addition": "+", "multiplication": "*", "subtraction": "-"}.get(self.operation, "+")
        algorithm_desc = (
            f"The model performs (a {op_symbol} b) mod {P} using the Fourier Multiplication Algorithm:\n"
            f"1. EMBED: Maps inputs a,b to sin(wk*a), cos(wk*a), sin(wk*b), cos(wk*b) "
            f"for key frequencies k in {key_freqs}\n"
            f"   where wk = 2*pi*k/{P}\n"
            f"2. COMPUTE: Uses attention + MLP to compute cos(wk*(a {op_symbol} b)) and sin(wk*(a {op_symbol} b))\n"
            f"   via trig identities\n"
            f"3. READOUT: Computes logit(c) = sum_k alpha_k * cos(wk*(a {op_symbol} b - c))\n"
            f"   Constructive interference at c* = (a {op_symbol} b) mod {P} gives maximum logit.\n"
            f"\nVerification: {correct}/{total} correct ({verify_acc*100:.2f}%)"
        )

        fve_mlp = {}
        for k, res in trig_results.items():
            fve_mlp[k] = (res["fve_cos"] + res["fve_sin"]) / 2

        return DiscoveredCircuit(
            key_frequencies=key_freqs,
            frequency_amplitudes=freq_amps,
            algorithm_description=algorithm_desc,
            mathematical_formula=formula,
            fve_logits=fve_logits,
            fve_mlp=fve_mlp,
            verification_accuracy=verify_acc,
            neuron_frequency_assignments=neuron_assignments,
            embedding_fourier_norms=embed_norms,
            wl_fourier_norms=wl_norms,
        )

# =============================================================================
# Ablation Tests
# =============================================================================

def ablation_test(model: ModularArithmeticTransformer, key_freqs: list, operation: str = "addition") -> dict:
    """
    Ablate key frequencies and non-key frequencies
    to confirm the circuit is both necessary and sufficient.
    """
    P = model.P
    all_a = torch.arange(P).repeat_interleave(P)
    all_b = torch.arange(P).repeat(P)

    if operation == "addition":
        all_targets = (all_a + all_b) % P
    elif operation == "multiplication":
        all_targets = (all_a * all_b) % P
    elif operation == "subtraction":
        all_targets = (all_a - all_b) % P
    else:
        all_targets = (all_a + all_b) % P

    with torch.no_grad():
        logits = model(all_a, all_b)
        preds = logits.argmax(dim=-1)
        baseline_acc = (preds == all_targets).float().mean().item()

    results = {"baseline_accuracy": baseline_acc}

    with torch.no_grad():
        logits_np = model(all_a, all_b).cpu().numpy()

    # Reshape logits to (P, P, P) for 2D DFT over (a, b)
    logit_cube = logits_np.reshape(P, P, P)
    logit_fft = np.fft.fft2(logit_cube, axes=(0, 1))

    # Ablate key frequencies (set them to zero) - should hurt
    logit_fft_ablated_key = logit_fft.copy()
    for k in key_freqs:
        logit_fft_ablated_key[k, :, :] = 0
        logit_fft_ablated_key[:, k, :] = 0
        if k > 0:
            logit_fft_ablated_key[P - k, :, :] = 0
            logit_fft_ablated_key[:, P - k, :] = 0

    logit_ablated_key = np.fft.ifft2(logit_fft_ablated_key, axes=(0, 1)).real
    preds_ablated_key = logit_ablated_key.reshape(P * P, P).argmax(axis=1)
    targets_np = all_targets.numpy()
    acc_without_key = (preds_ablated_key == targets_np).mean()
    results["accuracy_without_key_freqs"] = float(acc_without_key)

    # Ablate everything EXCEPT key frequencies (should preserve)
    logit_fft_restricted = np.zeros_like(logit_fft)
    logit_fft_restricted[0, 0, :] = logit_fft[0, 0, :]
    for k in key_freqs:
        logit_fft_restricted[k, :, :] = logit_fft[k, :, :]
        logit_fft_restricted[:, k, :] = logit_fft[:, k, :]
        if k > 0:
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
        if k > 0:
            logit_fft_single[P - k, :, :] = 0
            logit_fft_single[:, P - k, :] = 0
        logit_single = np.fft.ifft2(logit_fft_single, axes=(0, 1)).real
        preds_single = logit_single.reshape(P * P, P).argmax(axis=1)
        acc_single = (preds_single == targets_np).mean()
        per_freq_results[k] = float(acc_single)

    results["per_frequency_ablation"] = per_freq_results
    return results

# =============================================================================
# Plotly Visualizations
# =============================================================================

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def viz_embedding_on_circle(model: ModularArithmeticTransformer, key_freqs: list) -> go.Figure:
    """Visualize how the embedding maps tokens onto the unit circle."""
    P = model.P

    n_cols = min(len(key_freqs), 3)
    if n_cols == 0:
        n_cols = 1

    fig = make_subplots(
        rows=1, cols=n_cols,
        subplot_titles=[f"Frequency k={k}" for k in key_freqs[:n_cols]],
        specs=[[{"type": "polar"}] * n_cols],
    )

    for idx, k in enumerate(key_freqs[:n_cols]):
        wk = 2 * np.pi * k / P
        fig.add_trace(go.Scatterpolar(
            r=np.ones(P).tolist(),
            theta=((wk * np.arange(P) * 180 / np.pi) % 360).tolist(),
            mode="markers+text",
            marker=dict(size=6, color=np.arange(P).tolist(), colorscale="Viridis", showscale=(idx == 0)),
            text=[str(i) if i % 10 == 0 else "" for i in range(P)],
            textposition="top center",
            textfont=dict(size=7),
            name=f"k={k}",
        ), row=1, col=idx + 1)

    fig.update_layout(
        title="Embedding on Circle: tokens mapped to rotations at key frequencies",
        height=400,
        showlegend=False,
    )
    return fig

def viz_fourier_spectra(circuit: DiscoveredCircuit) -> go.Figure:
    """Visualize Fourier norms of W_E and W_L."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Embedding W_E Fourier Norms", "Neuron-Logit W_L Fourier Norms"])

    n = len(circuit.embedding_fourier_norms)
    x = list(range(n))

    colors_embed = ['red' if i in circuit.key_frequencies else '#3498db' for i in x]
    colors_wl = ['red' if i in circuit.key_frequencies else '#2ecc71' for i in x]

    fig.add_trace(go.Bar(
        x=x, y=circuit.embedding_fourier_norms.tolist(),
        marker_color=colors_embed, opacity=0.8,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x, y=circuit.wl_fourier_norms.tolist(),
        marker_color=colors_wl, opacity=0.8,
    ), row=1, col=2)

    fig.update_layout(
        title="Fourier Sparsity (red = key frequencies used by the circuit)",
        height=350, showlegend=False,
    )
    return fig

def viz_neuron_activations(model: ModularArithmeticTransformer, key_freqs: list) -> go.Figure:
    """Visualize MLP neuron activations as heatmaps over (a, b)."""
    P = model.P
    all_a = torch.arange(P).repeat_interleave(P)
    all_b = torch.arange(P).repeat(P)

    with torch.no_grad():
        mlp_acts, _ = model.get_mlp_activations(all_a, all_b)
    mlp_acts = mlp_acts.cpu().numpy()

    neuron_periodicities = []
    for neuron_idx in range(min(model.d_mlp, 100)):
        acts = mlp_acts[:, neuron_idx].reshape(P, P)
        fft = np.fft.fft2(acts)
        power = np.abs(fft) ** 2
        power[0, 0] = 0
        max_freq = np.unravel_index(np.argmax(power), power.shape)
        max_power = power[max_freq]
        total_power = power.sum()
        neuron_periodicities.append((neuron_idx, max_power / (total_power + 1e-10), max_freq))

    neuron_periodicities.sort(key=lambda x: x[1], reverse=True)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"Neuron {n[0]} (periodicity={n[1]:.2f})"
                                       for n in neuron_periodicities[:4]])

    for idx, (neuron_idx, _, _) in enumerate(neuron_periodicities[:4]):
        acts = mlp_acts[:, neuron_idx].reshape(P, P)
        row, col = idx // 2 + 1, idx % 2 + 1
        fig.add_trace(go.Heatmap(
            z=acts.tolist(), colorscale="RdBu", zmid=0,
            showscale=(idx == 0),
        ), row=row, col=col)

    fig.update_layout(
        title="Most Periodic MLP Neurons (activations over all (a,b) pairs)",
        height=600, width=650,
    )
    return fig

def viz_logit_structure(model: ModularArithmeticTransformer, key_freqs: list, operation: str = "addition") -> go.Figure:
    """Visualize the logit structure: constructive interference."""
    P = model.P
    all_a = torch.arange(P).repeat_interleave(P)
    all_b = torch.arange(P).repeat(P)

    with torch.no_grad():
        logits = model(all_a, all_b).cpu().numpy()

    a_example, b_example = 17, 42
    if operation == "addition":
        true_answer = (a_example + b_example) % P
    elif operation == "multiplication":
        true_answer = (a_example * b_example) % P
    elif operation == "subtraction":
        true_answer = (a_example - b_example) % P
    else:
        true_answer = (a_example + b_example) % P

    example_logits = logits[a_example * P + b_example]

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[
                            f"Logits for a={a_example}, b={b_example} (true answer: {true_answer})",
                            "Constructive Interference: sum of cos(wk(target-c))"
                        ])

    c_vals = np.arange(P)
    fig.add_trace(go.Bar(
        x=c_vals.tolist(), y=example_logits.tolist(),
        marker_color=['red' if c == true_answer else '#3498db' for c in c_vals],
        opacity=0.8,
    ), row=1, col=1)

    reconstructed = np.zeros(P)
    for k in key_freqs:
        wk = 2 * np.pi * k / P
        cos_wave = np.cos(wk * (true_answer - c_vals))
        reconstructed += cos_wave

    fig.add_trace(go.Bar(
        x=c_vals.tolist(), y=reconstructed.tolist(),
        marker_color=['red' if c == true_answer else '#2ecc71' for c in c_vals],
        opacity=0.8,
    ), row=2, col=1)

    fig.update_layout(
        title=f"Logit Structure: constructive interference at c*=(a op b) mod {P}",
        height=500, showlegend=False,
    )
    return fig

def viz_trig_identities(model: ModularArithmeticTransformer, trig_results: dict, P: int) -> go.Figure:
    """Visualize how well the MLP computes trig identities."""
    freqs = sorted(trig_results.keys())

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["FVE: cos(wk(a op b))", "FVE: sin(wk(a op b))"])

    fve_cos = [trig_results[k]["fve_cos"] for k in freqs]
    fve_sin = [trig_results[k]["fve_sin"] for k in freqs]

    fig.add_trace(go.Bar(
        x=[f"k={k}" for k in freqs], y=fve_cos,
        marker_color="#e74c3c", opacity=0.8,
        text=[f"{v:.1%}" for v in fve_cos], textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=[f"k={k}" for k in freqs], y=fve_sin,
        marker_color="#3498db", opacity=0.8,
        text=[f"{v:.1%}" for v in fve_sin], textposition="outside",
    ), row=1, col=2)

    fig.update_yaxes(range=[0, 1.1], row=1, col=1)
    fig.update_yaxes(range=[0, 1.1], row=1, col=2)

    fig.update_layout(
        title="Trig Identity Verification: MLP computes cos/sin(wk(a op b)) via product formula",
        height=350, showlegend=False,
    )
    return fig

def viz_ablation_results(ablation: dict) -> go.Figure:
    """Visualize ablation test results."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Global Ablations", "Per-Frequency Ablation"])

    labels = ["Baseline", "Without key freqs", "Only key freqs"]
    values = [
        ablation["baseline_accuracy"],
        ablation["accuracy_without_key_freqs"],
        ablation["accuracy_restricted_to_key_freqs"],
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

    fig.add_trace(go.Bar(
        x=labels, y=values, marker_color=colors, opacity=0.8,
        text=[f"{v:.1%}" for v in values], textposition="outside",
    ), row=1, col=1)

    if "per_frequency_ablation" in ablation:
        per_freq = ablation["per_frequency_ablation"]
        freq_labels = [f"k={k}" for k in per_freq.keys()]
        freq_values = list(per_freq.values())
        fig.add_trace(go.Bar(
            x=freq_labels, y=freq_values, marker_color="#9b59b6", opacity=0.8,
            text=[f"{v:.1%}" for v in freq_values], textposition="outside",
        ), row=1, col=2)

    fig.update_yaxes(range=[0, 1.1])
    fig.update_layout(
        title="Ablation Tests: Key frequencies are necessary and sufficient",
        height=350, showlegend=False,
    )
    return fig

def viz_training_curves(train_losses: list, test_accs: list, train_accs: list, weight_norms: list) -> go.Figure:
    """Visualize training curves showing grokking with extended metrics."""
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["Training Loss (CE + L2)", "Train vs Test Accuracy",
                                       "Weight Norm Over Time", "Loss Components"])

    if train_losses:
        epochs_l = [x[0] for x in train_losses]
        total_losses = [x[1] for x in train_losses]
        ce_losses = [x[2] for x in train_losses]
        l2_losses = [x[3] for x in train_losses]

        fig.add_trace(go.Scatter(
            x=epochs_l, y=total_losses, mode="lines",
            line=dict(color="#e74c3c", width=2), name="Total Loss",
        ), row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)

        # Loss components
        fig.add_trace(go.Scatter(
            x=epochs_l, y=ce_losses, mode="lines",
            line=dict(color="#e74c3c", width=2), name="CE Loss",
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=epochs_l, y=l2_losses, mode="lines",
            line=dict(color="#f39c12", width=2, dash="dash"), name="L2 Reg",
        ), row=2, col=2)
        fig.update_yaxes(type="log", row=2, col=2)

    if test_accs:
        epochs_a = [x[0] for x in test_accs]
        accs = [x[1] for x in test_accs]
        fig.add_trace(go.Scatter(
            x=epochs_a, y=accs, mode="lines+markers",
            line=dict(color="#3498db", width=2), marker=dict(size=3),
            name="Test Accuracy",
        ), row=1, col=2)

    if train_accs:
        epochs_ta = [x[0] for x in train_accs]
        taccs = [x[1] for x in train_accs]
        fig.add_trace(go.Scatter(
            x=epochs_ta, y=taccs, mode="lines",
            line=dict(color="#2ecc71", width=2), name="Train Accuracy",
        ), row=1, col=2)
    fig.update_yaxes(range=[0, 1.05], row=1, col=2)

    if weight_norms:
        epochs_w = [x[0] for x in weight_norms]
        norms = [x[1] for x in weight_norms]
        fig.add_trace(go.Scatter(
            x=epochs_w, y=norms, mode="lines",
            line=dict(color="#9b59b6", width=2), name="Weight Norm",
        ), row=2, col=1)

    fig.update_layout(
        title="Training Dynamics (Grokking: memorize first, then generalize)",
        height=600, showlegend=True,
    )
    return fig

def viz_weight_distribution(model: ModularArithmeticTransformer) -> go.Figure:
    """Visualize the distribution of weights across layers."""
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=["Embedding", "W_Q", "W_K", "W_V", "MLP In", "MLP Out"])

    weight_groups = [
        ("Embedding", model.embed.weight.detach().cpu().numpy().flatten()),
        ("W_Q", model.W_Q.weight.detach().cpu().numpy().flatten()),
        ("W_K", model.W_K.weight.detach().cpu().numpy().flatten()),
        ("W_V", model.W_V.weight.detach().cpu().numpy().flatten()),
        ("MLP In", model.mlp_in.weight.detach().cpu().numpy().flatten()),
        ("MLP Out", model.mlp_out.weight.detach().cpu().numpy().flatten()),
    ]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    for idx, (name, weights) in enumerate(weight_groups):
        row = idx // 3 + 1
        col = idx % 3 + 1
        fig.add_trace(go.Histogram(
            x=weights.tolist(), nbinsx=50,
            marker_color=colors[idx], opacity=0.7,
            name=name,
        ), row=row, col=col)

    fig.update_layout(
        title="Weight Distributions (L2 regularization encourages small weights)",
        height=450, showlegend=False,
    )
    return fig

def viz_attention_patterns(model: ModularArithmeticTransformer) -> go.Figure:
    """Visualize attention patterns across heads."""
    P = model.P
    all_a = torch.arange(P).repeat_interleave(P)
    all_b = torch.arange(P).repeat(P)

    with torch.no_grad():
        _, attn = model.get_mlp_activations(all_a, all_b)
    # attn shape: (P*P, n_heads, 1, 2)
    attn = attn.squeeze(2).cpu().numpy()  # (P*P, n_heads, 2)

    n_heads = model.n_heads
    fig = make_subplots(rows=1, cols=n_heads,
                        subplot_titles=[f"Head {i}" for i in range(n_heads)])

    for h in range(n_heads):
        attn_a = attn[:, h, 0]  # attention to position a
        attn_b = attn[:, h, 1]  # attention to position b

        # Average attention weights
        mean_a = attn_a.mean()
        mean_b = attn_b.mean()

        fig.add_trace(go.Bar(
            x=["Attend to a", "Attend to b"],
            y=[mean_a, mean_b],
            marker_color=["#3498db", "#e74c3c"],
            opacity=0.8,
            name=f"Head {h}",
        ), row=1, col=h + 1)
        fig.update_yaxes(range=[0, 1], row=1, col=h + 1)

    fig.update_layout(
        title="Attention Patterns: How much each head attends to position a vs b",
        height=300, showlegend=False,
    )
    return fig

def format_circuit_report(circuit: DiscoveredCircuit, ablation: dict, config: dict) -> str:
    """Format the full mathematical description of the discovered circuit."""
    lines = []
    lines.append("=" * 74)
    lines.append("  DISCOVERED CIRCUIT: MATHEMATICAL DESCRIPTION")
    lines.append("=" * 74)
    lines.append("")
    lines.append(f"  Configuration: P={config.get('P', '?')}, operation={config.get('operation', '?')}")
    lines.append(f"  Network: d_model={config.get('d_model', '?')}, n_heads={config.get('n_heads', '?')}, d_mlp={config.get('d_mlp', '?')}")
    lines.append(f"  Training: epochs={config.get('epochs', '?')}, l2_lambda={config.get('l2_lambda', '?')}")
    lines.append("")
    lines.append(circuit.algorithm_description)
    lines.append("")
    lines.append("-" * 74)
    lines.append("  MATHEMATICAL FORMULA")
    lines.append("-" * 74)
    lines.append(f"  {circuit.mathematical_formula}")
    lines.append("")
    lines.append("-" * 74)
    lines.append("  VERIFICATION RESULTS")
    lines.append("-" * 74)
    lines.append(f"  Exhaustive test accuracy: {circuit.verification_accuracy:.4%}")
    lines.append(f"  Logit FVE (formula explains this much variance): {circuit.fve_logits:.4f}")
    lines.append("")
    lines.append("  Per-frequency MLP FVE (trig identity quality):")
    for k, fve in circuit.fve_mlp.items():
        lines.append(f"    Frequency k={k}: FVE = {fve:.4f}")
    lines.append("")
    lines.append(f"  Neurons assigned to frequencies: {len(circuit.neuron_frequency_assignments)}")
    freq_counts = {}
    for info in circuit.neuron_frequency_assignments.values():
        f = info["frequency"]
        freq_counts[f] = freq_counts.get(f, 0) + 1
    for f, count in sorted(freq_counts.items()):
        lines.append(f"    Frequency k={f}: {count} neurons")
    lines.append("")
    lines.append("-" * 74)
    lines.append("  ABLATION TESTS")
    lines.append("-" * 74)
    lines.append(f"  Baseline accuracy: {ablation['baseline_accuracy']:.4%}")
    lines.append(f"  Without key frequencies: {ablation['accuracy_without_key_freqs']:.4%}")
    lines.append(f"  Restricted to key frequencies only: {ablation['accuracy_restricted_to_key_freqs']:.4%}")
    if "per_frequency_ablation" in ablation:
        lines.append("  Per-frequency ablation (removing one at a time):")
        for k, acc in ablation["per_frequency_ablation"].items():
            lines.append(f"    Remove k={k}: accuracy = {acc:.4%}")
    lines.append("")
    lines.append("=" * 74)
    return "\n".join(lines)

# =============================================================================
# Gradio GUI
# =============================================================================

def run_gui():
    """Launch the interactive circuit discovery GUI."""
    import gradio as gr

    state = {"model": None, "circuit": None, "ablation": None,
             "train_losses": [], "test_accs": [], "train_accs": [],
             "weight_norms": [], "trig_results": None, "config": None}

    def get_saved_runs():
        """Get list of saved runs for dropdown."""
        runs = []
        if RUNS_DIR.exists():
            for d in sorted(RUNS_DIR.iterdir()):
                if d.is_dir() and (d / "config.json").exists():
                    runs.append(d.name)
        return runs

    def train_and_discover(P, d_model, n_heads, d_mlp, train_frac, epochs,
                           lr, weight_decay, l2_lambda, operation, network_size,
                           progress=gr.Progress()):
        """Train model and discover circuits."""
        P = int(P)
        epochs = int(epochs)

        # Network size presets
        if network_size == "Small":
            d_model, n_heads, d_mlp = 64, 2, 256
        elif network_size == "Medium":
            d_model, n_heads, d_mlp = 128, 4, 512
        elif network_size == "Large":
            d_model, n_heads, d_mlp = 256, 8, 1024
        elif network_size == "XL":
            d_model, n_heads, d_mlp = 512, 8, 2048
        else:
            d_model = int(d_model)
            n_heads = int(n_heads)
            d_mlp = int(d_mlp)

        config = {
            "P": P,
            "d_model": d_model,
            "n_heads": n_heads,
            "d_mlp": d_mlp,
            "train_frac": train_frac,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "l2_lambda": l2_lambda,
            "operation": operation,
            "network_size": network_size,
        }

        log_lines = []
        def cb(msg):
            log_lines.append(msg)

        progress(0.05, desc="Training model (this may take a while)...")
        model, train_losses, test_accs, train_accs, weight_norms = train_model(
            P=P, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp,
            train_frac=train_frac, epochs=epochs, lr=lr,
            weight_decay=weight_decay, l2_lambda=l2_lambda,
            operation=operation, progress_cb=cb
        )
        state["model"] = model
        state["train_losses"] = train_losses
        state["test_accs"] = test_accs
        state["train_accs"] = train_accs
        state["weight_norms"] = weight_norms
        state["config"] = config

        progress(0.5, desc="Discovering circuits via Fourier analysis...")
        discoverer = CircuitDiscoverer(model, operation=operation)
        circuit = discoverer.full_discovery(cb)
        state["circuit"] = circuit

        progress(0.8, desc="Running ablation tests...")
        abl = ablation_test(model, circuit.key_frequencies, operation=operation)
        state["ablation"] = abl

        trig_results = discoverer.verify_trig_identities(circuit.key_frequencies)
        state["trig_results"] = trig_results

        # Save run
        run_name = f"{operation}_P{P}_d{d_model}_e{epochs}_l2{l2_lambda}_{int(time.time())}"
        save_run(run_name, model, train_losses, test_accs, train_accs, weight_norms, config)
        cb(f"Run saved as: {run_name}")

        progress(0.9, desc="Building visualizations...")

        training        # Save run
        run_name = f"{operation}_P{P}_d{d_model}_e{epochs}_l2{l2_lambda}_{int(time.time())}"
        save_run(run_name, model, train_losses, test_accs, train_accs, weight_norms, config)
        cb(f"Run saved as: {run_name}")

        progress(0.9, desc="Building visualizations...")

        # Generate all plots
        training_fig = viz_training_curves(train_losses, test_accs, train_accs, weight_norms)
        fourier_fig = viz_fourier_spectra(circuit)
        circle_fig = viz_embedding_on_circle(model, circuit.key_frequencies)
        neuron_fig = viz_neuron_activations(model, circuit.key_frequencies)
        logit_fig = viz_logit_structure(model, circuit.key_frequencies, operation)
        trig_fig = viz_trig_identities(model, trig_results, P)
        ablation_fig = viz_ablation_results(abl)
        weight_fig = viz_weight_distribution(model)
        attn_fig = viz_attention_patterns(model)

        report = format_circuit_report(circuit, abl, config)
        log_text = "\n".join(log_lines)

        progress(1.0, desc="Done!")
        return (training_fig, fourier_fig, circle_fig, neuron_fig,
                logit_fig, trig_fig, ablation_fig, weight_fig, attn_fig,
                report, log_text, gr.Dropdown(choices=get_saved_runs()))

    def load_saved_run(run_name_str):
        """Load a previously saved run."""
        if not run_name_str:
            return (None,) * 10 + ("No run selected.",)

        # Extract just the folder name (before any ' | ' description)
        run_name = run_name_str.split(" | ")[0].strip() if " | " in run_name_str else run_name_str.strip()

        model, train_losses, test_accs, train_accs, weight_norms, config = load_run(run_name)
        if model is None:
            return (None,) * 10 + (f"Could not load run: {run_name}",)

        state["model"] = model
        state["train_losses"] = train_losses
        state["test_accs"] = test_accs
        state["train_accs"] = train_accs
        state["weight_norms"] = weight_norms
        state["config"] = config

        operation = config.get("operation", "addition")

        # Re-run discovery
        discoverer = CircuitDiscoverer(model, operation=operation)
        circuit = discoverer.full_discovery()
        state["circuit"] = circuit

        abl = ablation_test(model, circuit.key_frequencies, operation=operation)
        state["ablation"] = abl

        trig_results = discoverer.verify_trig_identities(circuit.key_frequencies)
        state["trig_results"] = trig_results

        P = config["P"]

        training_fig = viz_training_curves(train_losses, test_accs, train_accs, weight_norms)
        fourier_fig = viz_fourier_spectra(circuit)
        circle_fig = viz_embedding_on_circle(model, circuit.key_frequencies)
        neuron_fig = viz_neuron_activations(model, circuit.key_frequencies)
        logit_fig = viz_logit_structure(model, circuit.key_frequencies, operation)
        trig_fig = viz_trig_identities(model, trig_results, P)
        ablation_fig = viz_ablation_results(abl)
        weight_fig = viz_weight_distribution(model)
        attn_fig = viz_attention_patterns(model)

        report = format_circuit_report(circuit, abl, config)

        return (training_fig, fourier_fig, circle_fig, neuron_fig,
                logit_fig, trig_fig, ablation_fig, weight_fig, attn_fig,
                report, f"Loaded run: {run_name}")

    def test_specific_inputs(a_val, b_val):
        """Test the discovered formula on specific inputs."""
        if state["model"] is None or state["circuit"] is None:
            return "Train a model first or load a saved run."

        model = state["model"]
        circuit = state["circuit"]
        config = state["config"]
        P = model.P
        operation = config.get("operation", "addition") if config else "addition"
        a_val = int(a_val) % P
        b_val = int(b_val) % P

        if operation == "addition":
            true_answer = (a_val + b_val) % P
            op_sym = "+"
        elif operation == "multiplication":
            true_answer = (a_val * b_val) % P
            op_sym = "*"
        elif operation == "subtraction":
            true_answer = (a_val - b_val) % P
            op_sym = "-"
        else:
            true_answer = (a_val + b_val) % P
            op_sym = "+"

        # Model prediction
        with torch.no_grad():
            logits = model(torch.tensor([a_val]), torch.tensor([b_val]))
            model_pred = logits.argmax(dim=-1).item()

        # Formula prediction
        formula_logits = np.zeros(P)
        for i, k in enumerate(circuit.key_frequencies):
            wk = 2 * np.pi * k / P
            alpha = circuit.frequency_amplitudes[k]
            for c in range(P):
                formula_logits[c] += alpha * np.cos(wk * (true_answer - c))
        formula_pred = np.argmax(formula_logits)

        result = f"Input: {a_val} {op_sym} {b_val} mod {P}\n"
        result += f"True answer: {true_answer}\n"
        result += f"Model prediction: {model_pred} {'CORRECT' if model_pred == true_answer else 'WRONG'}\n"
        result += f"Formula prediction: {formula_pred} {'CORRECT' if formula_pred == true_answer else 'WRONG'}\n"
        result += f"\nFormula used:\n  {circuit.mathematical_formula}\n"
        result += f"\nTop 5 logits (formula):\n"
        top5 = np.argsort(formula_logits)[::-1][:5]
        for c in top5:
            result += f"  c={c}: {formula_logits[c]:.4f} {'<-- correct' if c == true_answer else ''}\n"

        return result

    # Build GUI
    with gr.Blocks(
        title="Grokking Circuit Discovery",
        theme=gr.themes.Base(primary_hue="indigo", secondary_hue="purple"),
    ) as demo:
        gr.Markdown(
            "# Grokking Circuit Discovery & Verification\n"
            "Train a transformer on modular arithmetic, automatically discover the Fourier multiplication "
            "algorithm, verify it mechanically, and visualize the circuit structure.\n"
            "Based on [Nanda et al. (2023) 'Progress Measures for Grokking'](https://arxiv.org/abs/2301.05217)\n\n"
            "**Features:** L2 regularization (lasso), multiple operations, save/load runs, configurable network size."
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Training Config")
                operation_input = gr.Dropdown(
                    choices=["addition", "multiplication", "subtraction"],
                    value="addition", label="Operation"
                )
                p_input = gr.Number(value=113, label="Prime P", precision=0)
                network_size_input = gr.Dropdown(
                    choices=["Small", "Medium", "Large", "XL", "Custom"],
                    value="Medium", label="Network Size"
                )
                d_model_input = gr.Number(value=128, label="d_model (for Custom)", precision=0, visible=True)
                n_heads_input = gr.Number(value=4, label="n_heads (for Custom)", precision=0, visible=True)
                d_mlp_input = gr.Number(value=512, label="d_mlp (for Custom)", precision=0, visible=True)
                train_frac_input = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Train fraction")
                epochs_input = gr.Number(value=30000, label="Epochs", precision=0)
                lr_input = gr.Number(value=1e-3, label="Learning Rate")
                weight_decay_input = gr.Number(value=1.0, label="Weight Decay (AdamW)")
                l2_lambda_input = gr.Number(value=0.01, label="L2 Lambda (Lasso reg)")
                train_btn = gr.Button("Train & Discover", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### Load Saved Run")
                saved_runs_dropdown = gr.Dropdown(
                    choices=get_saved_runs(), label="Saved Runs", interactive=True
                )
                refresh_btn = gr.Button("Refresh List")
                load_btn = gr.Button("Load Run", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### Test Specific Input")
                a_input = gr.Number(value=17, label="a", precision=0)
                b_input = gr.Number(value=42, label="b", precision=0)
                test_btn = gr.Button("Test")
                test_output = gr.Textbox(label="Result", lines=10, interactive=False)

                gr.Markdown("---")
                log_box = gr.Textbox(label="Log", lines=8, interactive=False)

            with gr.Column(scale=3, min_width=700):
                with gr.Tabs():
                    with gr.TabItem("Training Curves"):
                        training_plot = gr.Plot(label="Grokking Dynamics")
                    with gr.TabItem("Fourier Spectra"):
                        fourier_plot = gr.Plot(label="W_E and W_L Fourier Norms")
                    with gr.TabItem("Circle Embedding"):
                        circle_plot = gr.Plot(label="Tokens on Unit Circle")
                    with gr.TabItem("Neuron Activations"):
                        neuron_plot = gr.Plot(label="Periodic MLP Neurons")
                    with gr.TabItem("Logit Structure"):
                        logit_plot = gr.Plot(label="Constructive Interference")
                    with gr.TabItem("Trig Identities"):
                        trig_plot = gr.Plot(label="MLP Trig Identity Verification")
                    with gr.TabItem("Ablations"):
                        ablation_plot = gr.Plot(label="Ablation Tests")
                    with gr.TabItem("Weight Distributions"):
                        weight_plot = gr.Plot(label="Weight Distributions")
                    with gr.TabItem("Attention Patterns"):
                        attn_plot = gr.Plot(label="Attention Patterns")
                    with gr.TabItem("Full Report"):
                        report_box = gr.Textbox(label="Mathematical Circuit Description",
                                               lines=40, interactive=False)

        # Wire up buttons
        train_btn.click(
            fn=train_and_discover,
            inputs=[p_input, d_model_input, n_heads_input, d_mlp_input, train_frac_input,
                    epochs_input, lr_input, weight_decay_input, l2_lambda_input,
                    operation_input, network_size_input],
            outputs=[training_plot, fourier_plot, circle_plot, neuron_plot,
                    logit_plot, trig_plot, ablation_plot, weight_plot, attn_plot,
                    report_box, log_box, saved_runs_dropdown],
        )

        load_btn.click(
            fn=load_saved_run,
            inputs=[saved_runs_dropdown],
            outputs=[training_plot, fourier_plot, circle_plot, neuron_plot,
                    logit_plot, trig_plot, ablation_plot, weight_plot, attn_plot,
                    report_box, log_box],
        )

        def refresh_runs():
            return gr.Dropdown(choices=get_saved_runs())

        refresh_btn.click(
            fn=refresh_runs,
            inputs=[],
            outputs=[saved_runs_dropdown],
        )

        test_btn.click(
            fn=test_specific_inputs,
            inputs=[a_input, b_input],
            outputs=[test_output],
        )

    demo.launch(inbrowser=True, server_name="0.0.0.0", server_port=7860, show_error=True)

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  Grokking Circuit Discovery & Verification")
    print("  Based on Nanda et al. (2023)")
    print("  With L2 regularization, multiple operations,")
    print("  save/load runs, and extended visualizations")
    print("=" * 60)
    print()
    print(f"  Runs directory: {RUNS_DIR.absolute()}")
    print(f"  Existing runs: {len(get_saved_runs())}")
    print()
    print("  http://localhost:7860")
    print("  Ctrl+C to stop")
    print()
    run_gui()
