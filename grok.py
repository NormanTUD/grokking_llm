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

The tool:
1. Trains (or loads) a small transformer on modular addition
2. Discovers key frequencies via Fourier analysis of weights
3. Verifies the circuit mechanically (ablations, FVE checks)
4. Visualizes embeddings on the circle, neuron activations, attention patterns
5. Provides mathematical descriptions of discovered circuits
6. Tests predictions exhaustively to confirm correctness

Usage:
    uv run circuit_extract.py
"""

import warnings
import math
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# Directory to save training logs
SAVE_DIR = "training_logs"
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# Minimal Transformer for Modular Arithmetic
# =============================================================================

class ModularAdditionTransformer(nn.Module):
    """
    1-layer transformer for modular addition, following Nanda et al. (2023).
    Input: "a b =" -> predicts (a+b) mod P
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
# Training with Live Progress
# =============================================================================

def train_model(P: int = 113, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512,
                train_frac: float = 0.3, epochs: int = 30000, lr: float = 1e-3,
                weight_decay: float = 1.0, progress_cb=None, progress=None) -> tuple:
    """Train a model on modular addition until it groks.

    Args:
        progress_cb: callback for log messages
        progress: Gradio progress object for live progress bar updates

    Returns:
        (model, train_losses, test_accs, metrics_table)
    """
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

            # Update Gradio progress bar with actual percentage and info
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
# Circuit Discovery: Fourier Analysis
# =============================================================================

@dataclass
class DiscoveredCircuit:
    """A mathematically exact description of a discovered circuit."""
    key_frequencies: list[int]
    frequency_amplitudes: dict  # freq -> amplitude
    algorithm_description: str
    mathematical_formula: str
    fve_logits: float  # fraction of variance explained
    fve_mlp: dict  # freq -> FVE for that frequency's trig identity
    verification_accuracy: float  # exhaustive test accuracy
    neuron_frequency_assignments: dict  # neuron_idx -> frequency
    embedding_fourier_norms: np.ndarray
    wl_fourier_norms: np.ndarray


class CircuitDiscoverer:
    """
    Automatically discover the Fourier multiplication circuit in a trained model.
    Implements the analysis from Nanda et al. (2023) Section 4.
    """

    def __init__(self, model: ModularAdditionTransformer):
        self.model = model
        self.P = model.P
        self.d_model = model.d_model
        self.d_mlp = model.d_mlp

    def _dft_matrix(self, N: int) -> np.ndarray:
        """Discrete Fourier Transform matrix."""
        n = np.arange(N)
        k = np.arange(N)
        return np.exp(-2j * np.pi * np.outer(k, n) / N) / np.sqrt(N)

    def _fourier_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Get real Fourier basis vectors: cos(2*pi*k*x/P) and sin(2*pi*k*x/P)."""
        P = self.P
        x = np.arange(P)
        cos_basis = np.zeros((P // 2 + 1, P))
        sin_basis = np.zeros((P // 2 + 1, P))
        for k in range(P // 2 + 1):
            cos_basis[k] = np.cos(2 * np.pi * k * x / P)
            sin_basis[k] = np.sin(2 * np.pi * k * x / P)
        return cos_basis, sin_basis

    def analyze_embedding_fourier(self) -> np.ndarray:
        """
        Compute Fourier norms of embedding matrix W_E.
        Per Section 4.1: W_E should be sparse in Fourier basis.
        """
        W_E = self.model.embed.weight[:self.P].detach().cpu().numpy()  # (P, d_model)
        P = self.P
        cos_basis, sin_basis = self._fourier_basis()

        norms = np.zeros(P // 2 + 1)
        for k in range(P // 2 + 1):
            cos_proj = cos_basis[k] @ W_E  # (d_model,)
            sin_proj = sin_basis[k] @ W_E  # (d_model,)
            norms[k] = np.sqrt(np.sum(cos_proj**2) + np.sum(sin_proj**2))

        return norms

    def analyze_neuron_logit_map(self) -> np.ndarray:
        """
        Compute Fourier norms of neuron-logit map W_L = W_U @ W_out.
        Per Section 4.2: W_L should be sparse in Fourier basis.
        """
        W_out = self.model.mlp_out.weight.detach().cpu().numpy()  # (d_model, d_mlp)
        W_U = self.model.unembed.weight.detach().cpu().numpy()    # (P, d_model)
        W_L = W_U @ W_out  # (P, d_mlp) - maps neurons to logits

        cos_basis, sin_basis = self._fourier_basis()
        P = self.P

        norms = np.zeros(P // 2 + 1)
        for k in range(P // 2 + 1):
            cos_proj = cos_basis[k] @ W_L  # (d_mlp,)
            sin_proj = sin_basis[k] @ W_L  # (d_mlp,)
            norms[k] = np.sqrt(np.sum(cos_proj**2) + np.sum(sin_proj**2))

        return norms

    def find_key_frequencies(self, n_freqs: int = 5) -> list[int]:
        """
        Identify key frequencies used by the model.
        These are frequencies with large norms in both W_E and W_L.
        """
        embed_norms = self.analyze_embedding_fourier()
        wl_norms = self.analyze_neuron_logit_map()

        # Combined score (both must be significant)
        combined = embed_norms * wl_norms
        # Exclude frequency 0 (constant)
        combined[0] = 0

        # Find top frequencies
        top_indices = np.argsort(combined)[::-1][:n_freqs]
        # Filter: only keep those significantly above noise
        threshold = combined[top_indices[0]] * 0.1
        key_freqs = [int(k) for k in top_indices if combined[k] > threshold]

        return key_freqs[:n_freqs]

    def verify_trig_identities(self, key_freqs: list[int]) -> dict:
        """
        Verify that MLP computes cos(wk(a+b)) and sin(wk(a+b)).
        """
        P = self.P
        model = self.model

        W_out = model.mlp_out.weight.detach().cpu().numpy()
        W_U = model.unembed.weight.detach().cpu().numpy()
        W_L = W_U @ W_out  # (P, d_mlp)

        cos_basis, sin_basis = self._fourier_basis()

        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        with torch.no_grad():
            mlp_acts, _ = model.get_mlp_activations(all_a, all_b)
        mlp_acts = mlp_acts.cpu().numpy()  # (P*P, d_mlp)

        results = {}
        for k in key_freqs:
            wk = 2 * np.pi * k / P

            cos_wk = cos_basis[k]  # (P,)
            sin_wk = sin_basis[k]  # (P,)

            u_k = W_L.T @ cos_wk  # (d_mlp,)
            v_k = W_L.T @ sin_wk  # (d_mlp,)
            u_k = u_k / (np.linalg.norm(u_k) + 1e-10)
            v_k = v_k / (np.linalg.norm(v_k) + 1e-10)

            cos_proj = mlp_acts @ u_k  # (P*P,)
            sin_proj = mlp_acts @ v_k  # (P*P,)

            a_vals = all_a.numpy()
            b_vals = all_b.numpy()
            true_cos = np.cos(wk * (a_vals + b_vals))
            true_sin = np.sin(wk * (a_vals + b_vals))

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
                "formula_cos": f"{alpha_cos:.1f} * cos(2pi*{k}*(a+b)/{P})",
                "formula_sin": f"{alpha_sin:.1f} * sin(2pi*{k}*(a+b)/{P})",
            }

        return results

    def verify_logit_approximation(self, key_freqs: list[int]) -> tuple[float, np.ndarray]:
        """
        Verify logits ~ sum_k alpha_k * cos(wk(a+b-c)).
        """
        P = self.P
        model = self.model

        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        with torch.no_grad():
            logits = model(all_a, all_b).cpu().numpy()  # (P*P, P)

        a_vals = all_a.numpy()
        b_vals = all_b.numpy()

        n_samples = P * P
        X = np.zeros((n_samples, P, len(key_freqs)))
        for i, k in enumerate(key_freqs):
            wk = 2 * np.pi * k / P
            for sample_idx in range(n_samples):
                a, b = a_vals[sample_idx], b_vals[sample_idx]
                for c in range(P):
                    X[sample_idx, c, i] = np.cos(wk * (a + b - c))

        X_flat = X.reshape(-1, len(key_freqs))
        y_flat = logits.flatten()

        alphas, _, _, _ = np.linalg.lstsq(X_flat, y_flat, rcond=None)

        y_pred = X_flat @ alphas
        ss_res = np.sum((y_flat - y_pred) ** 2)
        ss_tot = np.sum((y_flat - y_flat.mean()) ** 2)
        fve = 1.0 - ss_res / (ss_tot + 1e-10)

        return float(max(0, fve)), alphas

    def assign_neurons_to_frequencies(self, key_freqs: list[int]) -> dict:
        """
        Per Section 4.3: most neurons compute degree-2 polynomials of a single frequency.
        """
        P = self.P
        model = self.model

        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        with torch.no_grad():
            mlp_acts, _ = model.get_mlp_activations(all_a, all_b)
        mlp_acts = mlp_acts.cpu().numpy()  # (P*P, d_mlp)

        a_vals = all_a.numpy()
        b_vals = all_b.numpy()

        assignments = {}
        for neuron_idx in range(self.d_mlp):
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

    def exhaustive_verification(self, key_freqs: list[int], alphas: np.ndarray) -> tuple[float, int, int]:
        """
        Mechanically test the discovered circuit by computing predictions
        using ONLY the discovered formula and checking against ground truth.
        """
        P = self.P
        correct = 0
        total = P * P

        for a in range(P):
            for b in range(P):
                logits = np.zeros(P)
                for i, k in enumerate(key_freqs):
                    wk = 2 * np.pi * k / P
                    for c in range(P):
                        logits[c] += alphas[i] * np.cos(wk * (a + b - c))

                predicted = np.argmax(logits)
                true_answer = (a + b) % P
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

        # Build mathematical description
        P = self.P
        freq_amps = {k: float(alphas[i]) for i, k in enumerate(key_freqs)}

        formula_parts = []
        for i, k in enumerate(key_freqs):
            formula_parts.append(f"{alphas[i]:.2f} * cos(2pi*{k}*(a+b-c)/{P})")
        formula = "logit(c | a,b) = " + " + ".join(formula_parts)

        algorithm_desc = (
            f"The model performs modular addition (a+b) mod {P} using the Fourier Multiplication Algorithm:\n"
            f"1. EMBED: Maps inputs a,b to sin(wk*a), cos(wk*a), sin(wk*b), cos(wk*b) "
            f"for key frequencies k in {key_freqs}\n"
            f"   where wk = 2*pi*k/{P}\n"
            f"2. COMPUTE: Uses attention + MLP to compute cos(wk*(a+b)) and sin(wk*(a+b))\n"
            f"   via trig identities: cos(wk*(a+b)) = cos(wk*a)*cos(wk*b) - sin(wk*a)*sin(wk*b)\n"
            f"3. READOUT: Computes logit(c) = sum_k alpha_k * cos(wk*(a+b-c))\n"
            f"   Constructive interference at c* = (a+b) mod {P} gives maximum logit.\n"
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

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    }

    def load_saved_metrics():
        """Load previously saved training metrics if available."""
        csv_path = os.path.join(SAVE_DIR, "training_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
        return pd.DataFrame()

    def train_and_update(P, d_model, n_heads, d_mlp, train_frac, epochs, lr, weight_decay, progress=gr.Progress()):
        """Train model with live progress updates, returning plot and table."""
        logs = []

        def progress_cb(msg):
            logs.append(msg)

        model, train_losses, test_accs, metrics_table = train_model(
            P=int(P), d_model=int(d_model), n_heads=int(n_heads), d_mlp=int(d_mlp),
            train_frac=float(train_frac), epochs=int(epochs), lr=float(lr),
            weight_decay=float(weight_decay), progress_cb=progress_cb, progress=progress,
        )

        state["model"] = model
        state["train_losses"] = train_losses
        state["test_accs"] = test_accs
        state["metrics_table"] = metrics_table

        # Create plot
        fig = make_training_plot(train_losses, test_accs)

        # Create DataFrame for table display
        df = pd.DataFrame(metrics_table)

        # Log summary
        log_text = "\n".join(logs[-20:])  # Last 20 log lines

        return fig, df, log_text

    def discover_circuit(progress=gr.Progress()):
        """Run circuit discovery on the trained model."""
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
        gr.Markdown("# 🔬 Grokking Circuit Discovery & Verification Tool")
        gr.Markdown(
            "Train a small transformer on modular addition, observe grokking, "
            "and automatically discover the Fourier multiplication circuit."
        )

        with gr.Tabs():
            # ===== TAB 1: Training =====
            with gr.TabItem("🏋️ Training"):
                gr.Markdown("### Model & Training Configuration")
                with gr.Row():
                    with gr.Column(scale=1):
                        p_input = gr.Number(value=113, label="Prime P (modulus)", precision=0)
                        d_model_input = gr.Number(value=128, label="d_model", precision=0)
                        n_heads_input = gr.Number(value=4, label="n_heads", precision=0)
                        d_mlp_input = gr.Number(value=512, label="d_mlp", precision=0)
                    with gr.Column(scale=1):
                        train_frac_input = gr.Number(value=0.3, label="Train fraction")
                        epochs_input = gr.Number(value=30000, label="Max epochs", precision=0)
                        lr_input = gr.Number(value=1e-3, label="Learning rate")
                        wd_input = gr.Number(value=1.0, label="Weight decay")

                train_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")

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

            # ===== TAB 2: Circuit Discovery =====
            with gr.TabItem("🔍 Circuit Discovery"):
                gr.Markdown("### Discover the Fourier Multiplication Circuit")
                discover_btn = gr.Button("🔬 Discover Circuit", variant="primary", size="lg")

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

            # ===== TAB 3: Saved Training Data =====
            with gr.TabItem("📊 Saved Training Data"):
                gr.Markdown("### View Previously Saved Training Metrics")
                gr.Markdown(
                    f"Training metrics are automatically saved to `{SAVE_DIR}/training_metrics.csv` "
                    "after each training run."
                )
                load_btn = gr.Button("📂 Load Saved Data", variant="secondary")

                saved_table = gr.Dataframe(label="Saved Training Metrics", interactive=False)
                saved_plot = gr.Plot(label="Saved Training Curves")

                load_btn.click(
                    fn=view_saved_training_data,
                    inputs=[],
                    outputs=[saved_table, saved_plot],
                )

            # ===== TAB 4: About =====
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This Tool

                This tool implements the circuit discovery methodology from:

                **"Progress Measures for Grokking via Mechanistic Interpretability"**
                (Nanda et al., 2023)

                ### What is Grokking?

                Grokking is a phenomenon where a neural network first memorizes training data
                (achieving high train accuracy but low test accuracy), then suddenly generalizes
                after many more training steps.

                ### The Fourier Multiplication Algorithm

                The discovered circuit works as follows:
                1. **Embedding**: Maps inputs to Fourier components (sin/cos at key frequencies)
                2. **Attention**: Moves information from input positions to the output position
                3. **MLP**: Computes trig identities to get cos(wk*(a+b)) from cos(wk*a), sin(wk*a), etc.
                4. **Unembedding**: Converts back from Fourier space to logits via constructive interference

                ### Key Metrics
                - **FVE (Fraction of Variance Explained)**: How well the discovered formula explains model behavior
                - **Ablation accuracy**: Confirms key frequencies are necessary and sufficient
                - **Exhaustive verification**: Tests the formula on all P² possible inputs
                """)

    return demo


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    demo = build_gui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
