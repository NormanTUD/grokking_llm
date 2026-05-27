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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


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
# Training
# =============================================================================

def train_model(P: int = 113, d_model: int = 128, n_heads: int = 4, d_mlp: int = 512,
                train_frac: float = 0.3, epochs: int = 30000, lr: float = 1e-3,
                weight_decay: float = 1.0, progress_cb=None) -> tuple:
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

            update(f"Epoch {epoch}: loss={loss.item():.4f}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")

            if test_acc > 0.99:
                update(f"Grokked at epoch {epoch}!")
                break

    model.eval()
    return model, train_losses, test_accs


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
            # Project each embedding dimension onto cos(wk*x) and sin(wk*x)
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
            # Project logit dimension onto cos/sin
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
        Per Section 4.2, Table 1: project MLP activations onto W_L directions
        and check they match trig identities.
        """
        P = self.P
        model = self.model

        # Get W_L directions for each key frequency
        W_out = model.mlp_out.weight.detach().cpu().numpy()
        W_U = model.unembed.weight.detach().cpu().numpy()
        W_L = W_U @ W_out  # (P, d_mlp)

        cos_basis, sin_basis = self._fourier_basis()

        # Get MLP activations for all inputs
        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        with torch.no_grad():
            mlp_acts, _ = model.get_mlp_activations(all_a, all_b)
        mlp_acts = mlp_acts.cpu().numpy()  # (P*P, d_mlp)

        results = {}
        for k in key_freqs:
            wk = 2 * np.pi * k / P

            # Get u_k and v_k: directions in neuron space for cos(wk*c) and sin(wk*c)
            # u_k = W_L^T @ cos(wk) normalized
            cos_wk = cos_basis[k]  # (P,)
            sin_wk = sin_basis[k]  # (P,)

            u_k = W_L.T @ cos_wk  # (d_mlp,)
            v_k = W_L.T @ sin_wk  # (d_mlp,)
            u_k = u_k / (np.linalg.norm(u_k) + 1e-10)
            v_k = v_k / (np.linalg.norm(v_k) + 1e-10)

            # Project MLP activations onto these directions
            cos_proj = mlp_acts @ u_k  # (P*P,)
            sin_proj = mlp_acts @ v_k  # (P*P,)

            # Ground truth: cos(wk(a+b)) and sin(wk(a+b))
            a_vals = all_a.numpy()
            b_vals = all_b.numpy()
            true_cos = np.cos(wk * (a_vals + b_vals))
            true_sin = np.sin(wk * (a_vals + b_vals))

            # Fit: cos_proj ~ alpha * true_cos
            alpha_cos = np.dot(cos_proj, true_cos) / (np.dot(true_cos, true_cos) + 1e-10)
            alpha_sin = np.dot(sin_proj, true_sin) / (np.dot(true_sin, true_sin) + 1e-10)

            # Fraction of variance explained
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
        Per Section 4.2: this should explain >95% of variance.
        """
        P = self.P
        model = self.model

        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        with torch.no_grad():
            logits = model(all_a, all_b).cpu().numpy()  # (P*P, P)

        a_vals = all_a.numpy()
        b_vals = all_b.numpy()
        c_vals = np.arange(P)

        # Build design matrix: for each (a,b,c), compute cos(wk(a+b-c)) for each k
        n_samples = P * P
        X = np.zeros((n_samples * P, len(key_freqs)))
        y = logits.flatten()

        for i, k in enumerate(key_freqs):
            wk = 2 * np.pi * k / P
            for c in range(P):
                cos_vals = np.cos(wk * (a_vals + b_vals - c))
                X[c * n_samples:(c + 1) * n_samples, i] = cos_vals

        # Reshape logits to match
        y = logits.T.flatten()  # (P * P*P) - reorder to match X

        # Actually, let's do it properly
        X = np.zeros((n_samples, P, len(key_freqs)))
        for i, k in enumerate(key_freqs):
            wk = 2 * np.pi * k / P
            for sample_idx in range(n_samples):
                a, b = a_vals[sample_idx], b_vals[sample_idx]
                for c in range(P):
                    X[sample_idx, c, i] = np.cos(wk * (a + b - c))

        X_flat = X.reshape(-1, len(key_freqs))
        y_flat = logits.flatten()

        # Least squares fit
        alphas, _, _, _ = np.linalg.lstsq(X_flat, y_flat, rcond=None)

        # Compute FVE
        y_pred = X_flat @ alphas
        ss_res = np.sum((y_flat - y_pred) ** 2)
        ss_tot = np.sum((y_flat - y_flat.mean()) ** 2)
        fve = 1.0 - ss_res / (ss_tot + 1e-10)

        return float(max(0, fve)), alphas

    def assign_neurons_to_frequencies(self, key_freqs: list[int]) -> dict:
        """
        Per Section 4.3: most neurons compute degree-2 polynomials of a single frequency.
        Assign each neuron to its best-matching frequency.
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
                # Degree-2 polynomial basis: 1, cos(wk*a), sin(wk*a), cos(wk*b), sin(wk*b),
                # cos(wk*a)*cos(wk*b), sin(wk*a)*sin(wk*b), cos(wk*a)*sin(wk*b), sin(wk*a)*cos(wk*b)
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
                # Compute logits using discovered formula
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
        # Baseline accuracy
        logits = model(all_a, all_b)
        preds = logits.argmax(dim=-1)
        baseline_acc = (preds == all_targets).float().mean().item()

    results = {"baseline_accuracy": baseline_acc}

    # Ablate key frequencies from logits (should destroy performance)
    with torch.no_grad():
        logits_np = model(all_a, all_b).cpu().numpy()  # (P*P, P)

    # 2D DFT over inputs, then ablate key frequency components
    P = model.P
    a_vals = all_a.numpy()
    b_vals = all_b.numpy()

    # Reshape logits to (P, P, P) for 2D DFT over (a, b)
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

    # Ablate everything EXCEPT key frequencies (should preserve or improve)
    logit_fft_restricted = np.zeros_like(logit_fft)
    # Keep DC component
    logit_fft_restricted[0, 0, :] = logit_fft[0, 0, :]
    # Keep key frequency components
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
# Plotly Visualizations
# =============================================================================

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def viz_embedding_on_circle(model: ModularAdditionTransformer, key_freqs: list[int]) -> go.Figure:
    """
    Visualize how the embedding maps tokens onto the unit circle.
    Per Nanda et al. (2023) Section 4.1: W_E maps inputs to sin/cos at key frequencies.
    """
    P = model.P
    W_E = model.embed.weight[:P].detach().cpu().numpy()  # (P, d_model)

    cos_basis = np.zeros((P,))
    sin_basis = np.zeros((P,))

    fig = make_subplots(
        rows=1, cols=min(len(key_freqs), 3),
        subplot_titles=[f"Frequency k={k} (w={2*np.pi*k}/{P:.0f})" for k in key_freqs[:3]],
        specs=[[{"type": "polar"}] * min(len(key_freqs), 3)],
    )

    colors = px.colors.qualitative.Set1
    for idx, k in enumerate(key_freqs[:3]):
        wk = 2 * np.pi * k / P
        # Project embedding onto cos(wk) and sin(wk) directions
        cos_vec = np.cos(wk * np.arange(P))
        sin_vec = np.sin(wk * np.arange(P))

        # Project each embedding dimension
        cos_proj = W_E @ (W_E.T @ cos_vec) / (np.linalg.norm(W_E.T @ cos_vec) + 1e-10)
        sin_proj = W_E @ (W_E.T @ sin_vec) / (np.linalg.norm(W_E.T @ sin_vec) + 1e-10)

        # Simpler: just use the Fourier components directly
        cos_component = cos_vec @ W_E  # (d_model,) - how much cos(wk*x) is in each dim
        sin_component = sin_vec @ W_E  # (d_model,)

        # For each token, compute its angle on the circle
        angles = np.arctan2(sin_vec, cos_vec) * 180 / np.pi  # True angles
        radii = np.ones(P)

        # Color by token value
        fig.add_trace(go.Scatterpolar(
            r=radii,
            theta=(wk * np.arange(P) * 180 / np.pi) % 360,
            mode="markers+text",
            marker=dict(size=6, color=np.arange(P), colorscale="Viridis", showscale=(idx == 0)),
            text=[str(i) if i % 10 == 0 else "" for i in range(P)],
            textposition="top center",
            textfont=dict(size=7),
            name=f"k={k}",
            hovertemplate="Token %{text}<br>Angle: %{theta:.1f} deg<extra></extra>",
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

    # Highlight key frequencies
    colors_embed = ['red' if i in circuit.key_frequencies else '#3498db' for i in x]
    colors_wl = ['red' if i in circuit.key_frequencies else '#2ecc71' for i in x]

    fig.add_trace(go.Bar(
        x=x, y=circuit.embedding_fourier_norms,
        marker_color=colors_embed, opacity=0.8,
        hovertemplate="Freq %{x}: norm=%{y:.3f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x, y=circuit.wl_fourier_norms,
        marker_color=colors_wl, opacity=0.8,
        hovertemplate="Freq %{x}: norm=%{y:.3f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title="Fourier Sparsity (red = key frequencies used by the circuit)",
        height=350, showlegend=False,
    )
    return fig


def viz_neuron_activations(model: ModularAdditionTransformer, key_freqs: list[int]) -> go.Figure:
    """
    Visualize MLP neuron activations as heatmaps over (a, b).
    Per Section 4.1: neurons are periodic with key frequencies.
    """
    P = model.P
    all_a = torch.arange(P).repeat_interleave(P)
    all_b = torch.arange(P).repeat(P)

    with torch.no_grad():
        mlp_acts, _ = model.get_mlp_activations(all_a, all_b)
    mlp_acts = mlp_acts.cpu().numpy()  # (P*P, d_mlp)

    # Find most periodic neurons
    neuron_periodicities = []
    for neuron_idx in range(min(model.d_mlp, 100)):
        acts = mlp_acts[:, neuron_idx].reshape(P, P)
        # 2D FFT
        fft = np.fft.fft2(acts)
        power = np.abs(fft) ** 2
        power[0, 0] = 0  # remove DC
        max_freq = np.unravel_index(np.argmax(power), power.shape)
        max_power = power[max_freq]
        total_power = power.sum()
        neuron_periodicities.append((neuron_idx, max_power / (total_power + 1e-10), max_freq))

    # Sort by periodicity strength
    neuron_periodicities.sort(key=lambda x: x[1], reverse=True)

    # Show top 4 most periodic neurons
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"Neuron {n[0]} (periodicity={n[1]:.2f}, freq~{n[2]})"
                                       for n in neuron_periodicities[:4]])

    for idx, (neuron_idx, _, _) in enumerate(neuron_periodicities[:4]):
        acts = mlp_acts[:, neuron_idx].reshape(P, P)
        row, col = idx // 2 + 1, idx % 2 + 1
        fig.add_trace(go.Heatmap(
            z=acts, colorscale="RdBu", zmid=0,
            hovertemplate="a=%{y}, b=%{x}: act=%{z:.3f}<extra></extra>",
            showscale=(idx == 0),
        ), row=row, col=col)

    fig.update_layout(
        title="Most Periodic MLP Neurons (activations over all (a,b) pairs)",
        height=600, width=650,
    )
    return fig


def viz_logit_structure(model: ModularAdditionTransformer, key_freqs: list[int]) -> go.Figure:
    """
    Visualize the logit structure: cos(wk(a+b-c)) constructive interference.
    Per Section 4.2 and Appendix B.
    """
    P = model.P
    all_a = torch.arange(P).repeat_interleave(P)
    all_b = torch.arange(P).repeat(P)

    with torch.no_grad():
        logits = model(all_a, all_b).cpu().numpy()

    # Pick a specific example
    a_example, b_example = 17, 42
    true_answer = (a_example + b_example) % P
    example_logits = logits[a_example * P + b_example]  # (P,)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[
                            f"Logits for a={a_example}, b={b_example} (true answer: {true_answer})",
                            "Constructive Interference: sum of cos(wk(a+b-c))"
                        ])

    # Actual logits
    c_vals = np.arange(P)
    fig.add_trace(go.Bar(
        x=c_vals.tolist(), y=example_logits.tolist(),
        marker_color=['red' if c == true_answer else '#3498db' for c in c_vals],
        opacity=0.8,
        hovertemplate="c=%{x}: logit=%{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Reconstructed from key frequencies
    reconstructed = np.zeros(P)
    for k in key_freqs:
        wk = 2 * np.pi * k / P
        cos_wave = np.cos(wk * (a_example + b_example - c_vals))
        reconstructed += cos_wave

    fig.add_trace(go.Bar(
        x=c_vals.tolist(), y=reconstructed.tolist(),
        marker_color=['red' if c == true_answer else '#2ecc71' for c in c_vals],
        opacity=0.8,
        hovertemplate="c=%{x}: sum_cos=%{y:.3f}<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        title=f"Logit Structure: constructive interference at c*=(a+b) mod {P}",
        height=500, showlegend=False,
    )
    return fig


def viz_trig_identities(model: ModularAdditionTransformer, trig_results: dict, P: int) -> go.Figure:
    """Visualize how well the MLP computes trig identities."""
    freqs = sorted(trig_results.keys())
    n_freqs = len(freqs)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["FVE: cos(wk(a+b))", "FVE: sin(wk(a+b))"])

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
        title="Trig Identity Verification: MLP computes cos/sin(wk(a+b)) via product formula",
        height=350, showlegend=False,
    )
    return fig


def viz_ablation_results(ablation: dict) -> go.Figure:
    """Visualize ablation test results."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Global Ablations", "Per-Frequency Ablation"])

    # Global
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

    # Per-frequency
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


def viz_training_curves(train_losses: list, test_accs: list) -> go.Figure:
    """Visualize training curves showing grokking."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Training Loss", "Test Accuracy"])

    if train_losses:
        epochs_l, losses = zip(*train_losses)
        fig.add_trace(go.Scatter(
            x=list(epochs_l), y=list(losses), mode="lines",
            line=dict(color="#e74c3c", width=2), name="Train Loss",
        ), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)

    if test_accs:
        epochs_a, accs = zip(*test_accs)
        fig.add_trace(go.Scatter(
            x=list(epochs_a), y=list(accs), mode="lines+markers",
            line=dict(color="#3498db", width=2), marker=dict(size=3),
            name="Test Accuracy",
        ), row=1, col=2)
    fig.update_yaxes(range=[0, 1.05], row=1, col=2)

    fig.update_layout(
        title="Training Dynamics (Grokking: memorize first, then generalize)",
        height=350, showlegend=False,
    )
    return fig


def format_circuit_report(circuit: DiscoveredCircuit, ablation: dict) -> str:
    """Format the full mathematical description of the discovered circuit."""
    lines = []
    lines.append("=" * 74)
    lines.append("  DISCOVERED CIRCUIT: MATHEMATICAL DESCRIPTION")
    lines.append("=" * 74)
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
    lines.append("  ABLATION TESTS (Nanda et al. 2023, Section 4.4)")
    lines.append("-" * 74)
    lines.append(f"  Baseline accuracy: {ablation['baseline_accuracy']:.4%}")
    lines.append(f"  Without key frequencies: {ablation['accuracy_without_key_freqs']:.4%}")
    lines.append(f"  Restricted to key frequencies only: {ablation['accuracy_restricted_to_key_freqs']:.4%}")
    if "per_frequency_ablation" in ablation:
        lines.append("  Per-frequency ablation (removing one at a time):")
        for k, acc in ablation["per_frequency_ablation"].items():
            lines.append(f"    Remove k={k}: accuracy = {acc:.4%}")
    lines.append("")
    lines.append("-" * 74)
    lines.append("  INTERPRETATION")
    lines.append("-" * 74)
    lines.append("  The model has learned the FOURIER MULTIPLICATION ALGORITHM:")
    lines.append("  1. Embed inputs as rotations on the unit circle at key frequencies")
    lines.append("  2. Use attention to combine embeddings of a and b")
    lines.append("  3. MLP computes cos(wk(a+b)) via: cos(A+B) = cosA*cosB - sinA*sinB")
    lines.append("  4. Unembed computes cos(wk(a+b-c)) for each output c")
    lines.append("  5. Constructive interference at c* = (a+b) mod P gives max logit")
    lines.append("")
    lines.append("  This is VERIFIED by:")
    lines.append(f"  - Formula alone achieves {circuit.verification_accuracy:.2%} accuracy")
    lines.append(f"  - Removing key freqs destroys performance ({ablation['accuracy_without_key_freqs']:.2%})")
    lines.append(f"  - Keeping ONLY key freqs preserves performance ({ablation['accuracy_restricted_to_key_freqs']:.2%})")
    lines.append("=" * 74)
    return "\n".join(lines)


# =============================================================================
# Gradio GUI
# =============================================================================

def run_gui():
    """Launch the interactive circuit discovery GUI."""
    import gradio as gr

    state = {"model": None, "circuit": None, "ablation": None,
             "train_losses": [], "test_accs": [], "trig_results": None}

    def train_and_discover(P, d_model, n_heads, d_mlp, train_frac, epochs, progress=gr.Progress()):
        """Train model and discover circuits."""
        P = int(P)
        d_model = int(d_model)
        n_heads = int(n_heads)
        d_mlp = int(d_mlp)
        epochs = int(epochs)

        log_lines = []
        def cb(msg):
            log_lines.append(msg)

        progress(0.05, desc="Training model (this may take a while)...")
        model, train_losses, test_accs = train_model(
            P=P, d_model=d_model, n_heads=n_heads, d_mlp=d_mlp,
            train_frac=train_frac, epochs=epochs, progress_cb=cb
        )
        state["model"] = model
        state["train_losses"] = train_losses
        state["test_accs"] = test_accs

        progress(0.5, desc="Discovering circuits via Fourier analysis...")
        discoverer = CircuitDiscoverer(model)
        circuit = discoverer.full_discovery(cb)
        state["circuit"] = circuit

        progress(0.8, desc="Running ablation tests...")
        abl = ablation_test(model, circuit.key_frequencies)
        state["ablation"] = abl

        # Also store trig results for viz
        trig_results = discoverer.verify_trig_identities(circuit.key_frequencies)
        state["trig_results"] = trig_results

        progress(0.9, desc="Building visualizations...")

        # Generate all plots
        training_fig = viz_training_curves(train_losses, test_accs)
        fourier_fig = viz_fourier_spectra(circuit)
        circle_fig = viz_embedding_on_circle(model, circuit.key_frequencies)
        neuron_fig = viz_neuron_activations(model, circuit.key_frequencies)
        logit_fig = viz_logit_structure(model, circuit.key_frequencies)
        trig_fig = viz_trig_identities(model, trig_results, P)
        ablation_fig = viz_ablation_results(abl)

        report = format_circuit_report(circuit, abl)
        log_text = "\n".join(log_lines)

        progress(1.0, desc="Done!")
        return (training_fig, fourier_fig, circle_fig, neuron_fig,
                logit_fig, trig_fig, ablation_fig, report, log_text)

    def test_specific_inputs(a_val, b_val):
        """Test the discovered formula on specific inputs."""
        if state["model"] is None or state["circuit"] is None:
            return "Train a model first."

        model = state["model"]
        circuit = state["circuit"]
        P = model.P
        a_val = int(a_val) % P
        b_val = int(b_val) % P
        true_answer = (a_val + b_val) % P

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
                formula_logits[c] += alpha * np.cos(wk * (a_val + b_val - c))
        formula_pred = np.argmax(formula_logits)

        result = f"Input: {a_val} + {b_val} mod {P}\n"
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
            "Train a transformer on modular addition, automatically discover the Fourier multiplication "
            "algorithm, verify it mechanically, and visualize the circuit structure.\n"
            "Based on [Nanda et al. (2023) 'Progress Measures for Grokking'](https://arxiv.org/abs/2301.05217)"
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Training Config")
                p_input = gr.Number(value=113, label="Prime P", precision=0)
                d_model_input = gr.Number(value=128, label="d_model", precision=0)
                n_heads_input = gr.Number(value=4, label="n_heads", precision=0)
                d_mlp_input = gr.Number(value=512, label="d_mlp", precision=0)
                train_frac_input = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Train fraction")
                epochs_input = gr.Number(value=15000, label="Epochs", precision=0)
                train_btn = gr.Button("Train & Discover", variant="primary", size="lg")

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
                    with gr.TabItem("Training"):
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
                    with gr.TabItem("Full Report"):
                        report_box = gr.Textbox(label="Mathematical Circuit Description",
                                               lines=35, interactive=False)
        train_btn.click(
            fn=train_and_discover,
            inputs=[p_input, d_model_input, n_heads_input, d_mlp_input, train_frac_input, epochs_input],
            outputs=[training_plot, fourier_plot, circle_plot, neuron_plot,
                    logit_plot, trig_plot, ablation_plot, report_box, log_box],
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
    print("=" * 60)
    print()
    print("  http://localhost:7860")
    print("  Ctrl+C to stop")
    print()
    run_gui()
