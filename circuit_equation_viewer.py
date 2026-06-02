"""
Circuit Equation Viewer — Shows the FULL computation of (a + b) mod P
with both abstract equations and concrete variable substitution.

Traces every relevant weight matrix, activation, and intermediate value
from input to output. Filters to only show components that actually
contribute to the answer (key frequencies, assigned neurons).
"""

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional

# =============================================================================
# Core: Extract only the RELEVANT weights from the model
# =============================================================================

def extract_relevant_circuit_weights(model, key_frequencies: list[int],
                                      neuron_assignments: dict) -> dict:
    """
    Pull out ONLY the weight matrices and biases that participate in the
    modular addition circuit. This filters thousands of parameters down to
    the ones that actually do the computation.

    Adapted to work with ModularAdditionTransformer which uses:
      - model.W_Q, model.W_K, model.W_V, model.W_O: nn.Linear (no bias)
      - model.mlp_in: nn.Linear (with bias)
      - model.mlp_out: nn.Linear (with bias)
      - model.embed: nn.Embedding(P+1, d_model)
      - model.pos_embed: nn.Embedding(3, d_model)
      - model.unembed: nn.Linear(d_model, P, bias=False)
    """
    P = model.P
    d_model = model.d_model
    n_heads = model.n_heads
    d_head = d_model // n_heads

    with torch.no_grad():
        # Embeddings
        W_E = model.embed.weight[:P].cpu().numpy()          # (P, d_model)
        W_P = model.pos_embed.weight.cpu().numpy()          # (3, d_model)

        # Unembedding: nn.Linear(d_model, P, bias=False)
        # .weight shape is (P, d_model) — each row is the unembedding vector for class c
        W_U = model.unembed.weight.cpu().numpy()            # (P, d_model)

        # Attention weights: nn.Linear(d_model, d_model, bias=False)
        # .weight shape is (d_model, d_model) — we reshape to (n_heads, d_model, d_head)
        # Note: nn.Linear stores weight as (out_features, in_features)
        # So W_Q.weight is (d_model, d_model), and the forward does x @ W.T
        # i.e., Q = x @ W_Q.weight.T
        W_Q_full = model.W_Q.weight.cpu().numpy()  # (d_model, d_model)
        W_K_full = model.W_K.weight.cpu().numpy()  # (d_model, d_model)
        W_V_full = model.W_V.weight.cpu().numpy()  # (d_model, d_model)
        W_O_full = model.W_O.weight.cpu().numpy()  # (d_model, d_model)

        # Reshape into per-head matrices
        # nn.Linear: output = input @ weight.T
        # So Q = x @ W_Q.weight.T, and W_Q.weight.T is (d_model, d_model)
        # Per head: Q_h = x @ W_Q_h where W_Q_h is (d_model, d_head)
        # W_Q.weight.T reshaped: (d_model, n_heads, d_head)
        W_Q_T = W_Q_full.T  # (d_model, d_model) — this is what x gets multiplied by
        W_K_T = W_K_full.T  # (d_model, d_model)
        W_V_T = W_V_full.T  # (d_model, d_model)
        W_O_T = W_O_full.T  # (d_model, d_model)

        # Reshape to per-head: (d_model, n_heads, d_head) then transpose to (n_heads, d_model, d_head)
        W_Q = W_Q_T.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)  # (n_heads, d_model, d_head)
        W_K = W_K_T.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)  # (n_heads, d_model, d_head)
        W_V = W_V_T.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)  # (n_heads, d_model, d_head)

        # W_O: the model does attn_out.view(batch, 1, d_model) then W_O(attn_out)
        # W_O.weight is (d_model, d_model), forward: output = input @ W_O.weight.T
        # Per head contribution: head_h (d_head) gets projected by W_O_h (d_head, d_model)
        # W_O.weight.T is (d_model, d_model), reshape input side: (n_heads, d_head, d_model)
        W_O = W_O_T.reshape(n_heads, d_head, d_model)  # (n_heads, d_head, d_model)

        # MLP weights
        # mlp_in: nn.Linear(d_model, d_mlp) — weight is (d_mlp, d_model), bias is (d_mlp,)
        W_in = model.mlp_in.weight.cpu().numpy()    # (d_mlp, d_model)
        b_in = model.mlp_in.bias.cpu().numpy()      # (d_mlp,)

        # mlp_out: nn.Linear(d_mlp, d_model) — weight is (d_model, d_mlp), bias is (d_model,)
        W_out = model.mlp_out.weight.cpu().numpy()  # (d_model, d_mlp)
        b_out = model.mlp_out.bias.cpu().numpy()    # (d_model,)

    # Identify which neurons are relevant
    key_neuron_indices = sorted(int(idx) for idx in neuron_assignments.keys())

    # Precompute Fourier components for all tokens at key frequencies
    fourier_components = {}
    for k in key_frequencies:
        omega_k = 2 * np.pi * k / P
        cos_vals = np.cos(omega_k * np.arange(P))  # (P,)
        sin_vals = np.sin(omega_k * np.arange(P))  # (P,)
        fourier_components[k] = {"cos": cos_vals, "sin": sin_vals, "omega": omega_k}

    return {
        "W_E": W_E,
        "W_P": W_P,
        "W_Q": W_Q,
        "W_K": W_K,
        "W_V": W_V,
        "W_O": W_O,
        "W_in": W_in,
        "b_in": b_in,
        "W_out": W_out,
        "b_out": b_out,
        "W_U": W_U,
        "P": P,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_head": d_head,
        "d_mlp": W_in.shape[0],
        "key_frequencies": key_frequencies,
        "key_neuron_indices": key_neuron_indices,
        "neuron_assignments": neuron_assignments,
        "fourier_components": fourier_components,
    }

# =============================================================================
# Full Forward Pass with Equation Annotations
# =============================================================================

def trace_full_computation(model, a: int, b: int, key_frequencies: list[int],
                           neuron_assignments: dict) -> dict:
    """
    Run (a, b) through the model and record EVERY intermediate value
    that participates in the circuit. Returns a structured dict with:
      - Each step's abstract equation (LaTeX string)
      - The concrete values after substituting a, b
      - The actual tensor values from the model
      - Comparison between the "ideal" Fourier formula and actual model output

    This is the main function that powers the interactive viewer.
    """
    P = model.P
    d_model = model.d_model
    n_heads = model.n_heads
    d_head = d_model // n_heads
    correct = (a + b) % P

    weights = extract_relevant_circuit_weights(model, key_frequencies, neuron_assignments)

    # =========================================================================
    # STEP 1: EMBEDDING
    # =========================================================================
    W_E = weights["W_E"]
    W_P = weights["W_P"]

    x0 = W_E[a] + W_P[0]   # position 0: token a
    x1 = W_E[b] + W_P[1]   # position 1: token b
    x2 = W_E[P] + W_P[2] if hasattr(model, '_eq_token') else W_P[2]  # position 2: "=" token

    # For the "=" token, the model uses index P (the 114th row of embedding)
    # Actually check the model's forward to see how it handles the = token
    # In ModularAdditionTransformer, the = token is typically index P
    a_tensor = torch.tensor([a])
    b_tensor = torch.tensor([b])

    with torch.no_grad():
        logits_full, activations = model.forward_with_hooks(a_tensor, b_tensor)

    embed_actual = activations["embed"][0].cpu().numpy()  # (3, d_model)
    x0_actual = embed_actual[0]
    x1_actual = embed_actual[1]
    x2_actual = embed_actual[2]

    # Fourier decomposition of embeddings at key frequencies
    fourier_embed = {}
    for k in key_frequencies:
        omega_k = 2 * np.pi * k / P
        fourier_embed[k] = {
            "cos_a": np.cos(omega_k * a),
            "sin_a": np.sin(omega_k * a),
            "cos_b": np.cos(omega_k * b),
            "sin_b": np.sin(omega_k * b),
            "cos_apb": np.cos(omega_k * (a + b)),
            "sin_apb": np.sin(omega_k * (a + b)),
            "omega": omega_k,
        }

    step1 = {
        "name": "Embedding",
        "abstract_equation": (
            r"$\mathbf{x}_0 = W_E[a] + W_P[0]$, "
            r"$\mathbf{x}_1 = W_E[b] + W_P[1]$, "
            r"$\mathbf{x}_2 = W_E[\text{=}] + W_P[2]$"
        ),
        "concrete_equation": (
            f"$\\mathbf{{x}}_0 = W_E[{a}] + W_P[0]$, "
            f"$\\mathbf{{x}}_1 = W_E[{b}] + W_P[1]$"
        ),
        "fourier_decomposition": fourier_embed,
        "values": {
            "x0": x0_actual,
            "x1": x1_actual,
            "x2": x2_actual,
            "x0_norm": np.linalg.norm(x0_actual),
            "x1_norm": np.linalg.norm(x1_actual),
        },
    }

    # =========================================================================
    # STEP 2: ATTENTION
    # =========================================================================
    attn_weights_actual = activations["attn_weights"][0].cpu().numpy()  # (n_heads, 1, 2)
    # attn_weights_actual[h, 0, 0] = attention from "=" to "a"
    # attn_weights_actual[h, 0, 1] = attention from "=" to "b"

    attn_head_outputs = []
    for h in range(n_heads):
        head_out = activations[f"attn_head_{h}"][0, 0].cpu().numpy()  # (d_head,)
        attn_head_outputs.append(head_out)

    attn_combined = activations["attn_out"][0, 0].cpu().numpy()  # (d_model,)

    step2 = {
        "name": "Attention",
        "abstract_equation": (
            r"$A_0^{(h)} = \text{softmax}\left(\frac{\mathbf{q}^{(h)} \cdot \mathbf{k}_0^{(h)\top}}{\sqrt{d_h}}, "
            r"\frac{\mathbf{q}^{(h)} \cdot \mathbf{k}_1^{(h)\top}}{\sqrt{d_h}}\right)_0$"
            "\n\n"
            r"$\text{Attn}(\mathbf{x}) = \sum_h \left(A_0^{(h)} \mathbf{v}_0^{(h)} + A_1^{(h)} \mathbf{v}_1^{(h)}\right) W_O^{(h)}$"
        ),
        "concrete_values": {
            "attention_weights_per_head": {
                h: {"to_a": float(attn_weights_actual[h, 0, 0]),
                     "to_b": float(attn_weights_actual[h, 0, 1])}
                for h in range(n_heads)
            },
            "head_outputs": attn_head_outputs,
            "combined_output": attn_combined,
            "combined_norm": float(np.linalg.norm(attn_combined)),
        },
        "interpretation": (
            f"Attention weights ≈ [0.5, 0.5] means the '=' position "
            f"receives equal info from both a={a} and b={b}. "
            f"Actual: " + ", ".join(
                f"Head {h}: [{attn_weights_actual[h,0,0]:.3f}, {attn_weights_actual[h,0,1]:.3f}]"
                for h in range(n_heads)
            )
        ),
    }

    # =========================================================================
    # STEP 3: RESIDUAL MID (input to MLP)
    # =========================================================================
    residual_mid = activations["residual_mid"][0, 0].cpu().numpy()  # (d_model,)

    step3_residual = {
        "name": "Residual Stream (Mid)",
        "abstract_equation": r"$\mathbf{r}_{\text{mid}} = \mathbf{x}_2 + \text{Attn}(\mathbf{x})$",
        "concrete_equation": (
            f"$\\mathbf{{r}}_{{\\text{{mid}}}} = \\mathbf{{x}}_2 + \\text{{Attn}}({a}, {b})$"
        ),
        "values": {
            "residual_mid": residual_mid,
            "norm": float(np.linalg.norm(residual_mid)),
        },
    }

    # =========================================================================
    # STEP 4: MLP — THE KEY COMPUTATION
    # =========================================================================
    mlp_pre = activations["mlp_pre"][0, 0].cpu().numpy()    # (d_mlp,)
    mlp_hidden = activations["mlp_hidden"][0, 0].cpu().numpy()  # (d_mlp,) after ReLU
    mlp_out = activations["mlp_out"][0, 0].cpu().numpy()    # (d_model,)

    # For each KEY neuron, show what it computes
    key_neuron_details = []
    for neuron_idx_str, info in neuron_assignments.items():
        neuron_idx = int(neuron_idx_str)
        freq = info["frequency"]
        omega_k = 2 * np.pi * freq / P

        cos_a = np.cos(omega_k * a)
        sin_a = np.sin(omega_k * a)
        cos_b = np.cos(omega_k * b)
        sin_b = np.sin(omega_k * b)

        # The ideal value this neuron should compute
        ideal_cos_apb = np.cos(omega_k * (a + b))
        ideal_sin_apb = np.sin(omega_k * (a + b))

        # Trig identity: cos(a)cos(b) - sin(a)sin(b) = cos(a+b)
        trig_product = cos_a * cos_b - sin_a * sin_b

        actual_pre = mlp_pre[neuron_idx]
        actual_post = mlp_hidden[neuron_idx]

        key_neuron_details.append({
            "neuron_idx": neuron_idx,
            "frequency": freq,
            "omega_k": omega_k,
            "cos_a": cos_a,
            "sin_a": sin_a,
            "cos_b": cos_b,
            "sin_b": sin_b,
            "ideal_cos_apb": ideal_cos_apb,
            "trig_product": trig_product,
            "actual_pre_activation": actual_pre,
            "actual_post_relu": actual_post,
            "correlation_with_ideal": float(np.sign(actual_pre) * np.sign(trig_product))
                if abs(trig_product) > 1e-6 else 0.0,
        })

    # Summary: how many neurons fire, grouped by frequency
    neurons_by_freq = {}
    for detail in key_neuron_details:
        k = detail["frequency"]
        if k not in neurons_by_freq:
            neurons_by_freq[k] = {"firing": 0, "total": 0, "details": []}
        neurons_by_freq[k]["total"] += 1
        if detail["actual_post_relu"] > 0:
            neurons_by_freq[k]["firing"] += 1
        neurons_by_freq[k]["details"].append(detail)

    step4_mlp = {
        "name": "MLP (Trig Identity Computation)",
        "abstract_equation": (
            r"$z_n = \mathbf{r}_{\text{mid}} \cdot W_{\text{in}}[n, :] + b_{\text{in}}[n]$"
            "\n\n"
            r"$\approx \gamma_n \left[\cos(\omega_k a)\cos(\omega_k b) - \sin(\omega_k a)\sin(\omega_k b)\right]$"
            "\n\n"
            r"$= \gamma_n \cos(\omega_k(a+b))$"
            "\n\n"
            r"$m_n = \text{ReLU}(z_n)$"
        ),
        "concrete_equation": (
            f"For a={a}, b={b}:\n\n" +
            "\n".join(
                f"  Neuron {d['neuron_idx']} (k={d['frequency']}): "
                f"cos({d['omega_k']:.4f}·{a})·cos({d['omega_k']:.4f}·{b}) - "
                f"sin({d['omega_k']:.4f}·{a})·sin({d['omega_k']:.4f}·{b}) = "
                f"{d['cos_a']:.4f}·{d['cos_b']:.4f} - ({d['sin_a']:.4f})·({d['sin_b']:.4f}) = "
                f"{d['trig_product']:.4f} ≈ cos(ω_{d['frequency']}·{a+b}) = {d['ideal_cos_apb']:.4f}"
                for d in key_neuron_details[:10]  # Show first 10
            )
        ),
        "neurons_by_frequency": neurons_by_freq,
        "key_neuron_details": key_neuron_details,
        "total_active_neurons": int((mlp_hidden > 0).sum()),
        "key_active_neurons": sum(1 for d in key_neuron_details if d["actual_post_relu"] > 0),
        "mlp_output_norm": float(np.linalg.norm(mlp_out)),
    }

    # =========================================================================
    # STEP 5: FINAL RESIDUAL
    # =========================================================================
    residual_final = activations["residual_final"][0, 0].cpu().numpy()  # (d_model,)

    step5_residual = {
        "name": "Final Residual Stream",
        "abstract_equation": r"$\mathbf{r}_{\text{final}} = \mathbf{r}_{\text{mid}} + \text{MLP}(\mathbf{r}_{\text{mid}})$",
        "values": {
            "residual_final": residual_final,
            "norm": float(np.linalg.norm(residual_final)),
        },
    }

    # =========================================================================
    # STEP 6: UNEMBEDDING (LOGITS)
    # =========================================================================
    logits = activations["logits"][0].cpu().numpy()  # (P,)

    # Compute the IDEAL logits from the Fourier formula
    ideal_logits = np.zeros(P)
    for k in key_frequencies:
        omega_k = 2 * np.pi * k / P
        for c in range(P):
            ideal_logits[c] += np.cos(omega_k * (a + b - c))

    # Normalize ideal logits to match scale
    if np.std(ideal_logits) > 0:
        scale = np.std(logits) / np.std(ideal_logits)
        ideal_logits_scaled = ideal_logits * scale
    else:
        ideal_logits_scaled = ideal_logits

    # Correlation between actual and ideal
    correlation = np.corrcoef(logits, ideal_logits)[0, 1] if np.std(ideal_logits) > 0 else 0.0

    # Top predictions
    top_k = 5
    top_indices = np.argsort(logits)[-top_k:][::-1]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    step6_logits = {
        "name": "Unembedding (Logits)",
        "abstract_equation": (
            r"$\text{Logit}(c) = \mathbf{r}_{\text{final}} \cdot W_U[:, c]$"
            "\n\n"
            r"$\approx \sum_{k \in \mathcal{K}} \alpha_k \cos\!\left(\frac{2\pi k(a+b-c)}{P}\right)$"
        ),
        "concrete_equation": (
            f"$\\text{{Logit}}(c) \\approx \\sum_{{k \\in {{{', '.join(str(k) for k in key_frequencies)}}}}} "
            f"\\alpha_k \\cos\\!\\left(\\frac{{2\\pi k({a}+{b}-c)}}{{{P}}}\\right)$"
            f"\n\nFor correct answer c={correct}: "
            f"$\\cos(0) = 1$ for ALL k → constructive interference!"
            f"\n\nActual logit[{correct}] = {logits[correct]:.4f} (rank: {list(np.argsort(logits)[::-1]).index(correct)+1})"
        ),
        "values": {
            "logits": logits,
            "ideal_logits": ideal_logits,
            "ideal_logits_scaled": ideal_logits_scaled,
            "correlation_actual_vs_ideal": float(correlation),
            "predicted": int(logits.argmax()),
            "correct": correct,
            "is_correct": bool(logits.argmax() == correct),
            "top_k": [(int(idx), float(logits[idx]), float(probs[idx])) for idx in top_indices],
            "logit_at_correct": float(logits[correct]),
            "prob_at_correct": float(probs[correct]),
        },
        "interference_demo": {
            "at_correct": {k: float(np.cos(2*np.pi*k*(a+b-correct)/P))
                           for k in key_frequencies},
            "at_wrong_example": {k: float(np.cos(2*np.pi*k*(a+b-(correct+1)%P)/P))
                                  for k in key_frequencies},
        },
    }

    # =========================================================================
    # ASSEMBLE FULL TRACE
    # =========================================================================
    return {
        "input": {"a": a, "b": b, "P": P, "correct": correct},
        "steps": [step1, step2, step3_residual, step4_mlp, step5_residual, step6_logits],
        "summary": {
            "predicted": int(logits.argmax()),
            "correct": correct,
            "is_correct": bool(logits.argmax() == correct),
            "confidence": float(probs[correct]),
            "correlation_with_formula": float(correlation),
        },
        "weights_info": {
            "key_frequencies": key_frequencies,
            "n_key_neurons": len(neuron_assignments),
            "d_model": d_model,
            "n_heads": n_heads,
            "d_mlp": weights["d_mlp"],
        },
    }


# =============================================================================
# Visualization: Full Equation Flow as Plotly Figure
# =============================================================================

def make_equation_flow_figure(trace: dict) -> go.Figure:
    """
    Create a comprehensive multi-panel figure showing the full computation
    with equations and actual values side by side.
    """
    a = trace["input"]["a"]
    b = trace["input"]["b"]
    P = trace["input"]["P"]
    correct = trace["input"]["correct"]
    steps = trace["steps"]

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            f"① Fourier Components of a={a}, b={b}",
            f"② Attention Weights (= → a, b)",
            f"③ MLP Neurons by Frequency (pre-ReLU)",
            f"④ MLP Neurons (post-ReLU, active only)",
            f"⑤ Trig Identity Verification",
            f"⑥ Logits: Actual vs Ideal Formula",
            f"⑦ Constructive Interference at c={correct}",
            f"⑧ Probability Distribution",
        ),
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    # --- Panel 1: Fourier components ---
    fourier = steps[0]["fourier_decomposition"]
    freqs = sorted(fourier.keys())
    cos_a_vals = [fourier[k]["cos_a"] for k in freqs]
    sin_a_vals = [fourier[k]["sin_a"] for k in freqs]
    cos_b_vals = [fourier[k]["cos_b"] for k in freqs]
    sin_b_vals = [fourier[k]["sin_b"] for k in freqs]

    fig.add_trace(go.Bar(name=f"cos(ωk·{a})", x=[f"k={k}" for k in freqs],
                         y=cos_a_vals, marker_color="blue", opacity=0.7), row=1, col=1)
    fig.add_trace(go.Bar(name=f"sin(ωk·{a})", x=[f"k={k}" for k in freqs],
                         y=sin_a_vals, marker_color="lightblue", opacity=0.7), row=1, col=1)
    fig.add_trace(go.Bar(name=f"cos(ωk·{b})", x=[f"k={k}" for k in freqs],
                         y=cos_b_vals, marker_color="red", opacity=0.7), row=1, col=1)
    fig.add_trace(go.Bar(name=f"sin(ωk·{b})", x=[f"k={k}" for k in freqs],
                         y=sin_b_vals, marker_color="lightsalmon", opacity=0.7), row=1, col=1)

    # --- Panel 2: Attention weights ---
    attn_data = steps[1]["concrete_values"]["attention_weights_per_head"]
    heads = sorted(attn_data.keys())
    fig.add_trace(go.Bar(name="Attn to a", x=[f"Head {h}" for h in heads],
                         y=[attn_data[h]["to_a"] for h in heads],
                         marker_color="blue"), row=1, col=2)
    fig.add_trace(go.Bar(name="Attn to b", x=[f"Head {h}" for h in heads],
                         y=[attn_data[h]["to_b"] for h in heads],
                         marker_color="orange"), row=1, col=2)

    # --- Panel 3: MLP pre-activations for key neurons ---
    mlp_step = steps[3]
    neuron_details = mlp_step["key_neuron_details"]

    # Group by frequency, show pre-activation values
    for k in sorted(mlp_step["neurons_by_frequency"].keys()):
        freq_neurons = [d for d in neuron_details if d["frequency"] == k]
        pre_vals = [d["actual_pre_activation"] for d in freq_neurons]
        indices = [d["neuron_idx"] for d in freq_neurons]
        fig.add_trace(go.Bar(
            name=f"k={k} (pre-ReLU)",
            x=[str(i) for i in indices[:20]],  # limit display
            y=pre_vals[:20],
            opacity=0.7,
        ), row=2, col=1)

    # --- Panel 4: Post-ReLU (active neurons only) ---
    active_neurons = [(d["neuron_idx"], d["actual_post_relu"], d["frequency"])
                      for d in neuron_details if d["actual_post_relu"] > 0]
    if active_neurons:
        indices, vals, freqs_list = zip(*active_neurons)
        colors = [f"hsl({(f * 60) % 360}, 70%, 50%)" for f in freqs_list]
        fig.add_trace(go.Bar(
            x=[str(i) for i in indices[:30]],
            y=list(vals[:30]),
            marker_color=colors[:30],
            name="Active neurons",
            hovertemplate="Neuron %{x}<br>Value: %{y:.4f}<extra></extra>",
        ), row=2, col=2)

    # --- Panel 5: Trig identity verification ---
    # For each frequency, plot ideal cos(ωk(a+b)) vs actual neuron average
    freq_ideal = []
    freq_actual_avg = []
    freq_labels = []
    for k in sorted(mlp_step["neurons_by_frequency"].keys()):
        omega_k = 2 * np.pi * k / P
        ideal = np.cos(omega_k * (a + b))
        freq_neurons = mlp_step["neurons_by_frequency"][k]["details"]
        # Average of pre-activations (normalized)
        pre_vals = [d["actual_pre_activation"] for d in freq_neurons]
        if pre_vals:
            avg_pre = np.mean(pre_vals)
            # Normalize to [-1, 1] range for comparison
            max_abs = max(abs(v) for v in pre_vals) if pre_vals else 1.0
            normalized_avg = avg_pre / (max_abs + 1e-10)
        else:
            normalized_avg = 0.0
        freq_ideal.append(ideal)
        freq_actual_avg.append(normalized_avg)
        freq_labels.append(f"k={k}")

    fig.add_trace(go.Scatter(
        x=freq_labels, y=freq_ideal, mode="markers+lines",
        name="Ideal cos(ωk(a+b))", marker=dict(size=12, symbol="star"),
        line=dict(color="green", width=2),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=freq_labels, y=freq_actual_avg, mode="markers+lines",
        name="Actual (normalized avg)", marker=dict(size=10),
        line=dict(color="purple", width=2, dash="dash"),
    ), row=3, col=1)

    # --- Panel 6: Actual vs Ideal logits ---
    logit_step = steps[5]
    logits = logit_step["values"]["logits"]
    ideal_logits = logit_step["values"]["ideal_logits_scaled"]

    # Show a window around the correct answer
    window = 20
    start = max(0, correct - window)
    end = min(P, correct + window)
    x_range = list(range(start, end))

    fig.add_trace(go.Scatter(
        x=x_range, y=logits[start:end], mode="lines",
        name="Actual logits", line=dict(color="blue", width=2),
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=x_range, y=ideal_logits[start:end], mode="lines",
        name="Ideal (formula)", line=dict(color="red", width=2, dash="dash"),
    ), row=3, col=2)
    fig.add_trace(go.Scatter(
        x=[correct], y=[logits[correct]], mode="markers",
        marker=dict(size=15, color="green", symbol="star"),
        name=f"Correct: c={correct}",
    ), row=3, col=2)

    # --- Panel 7: Constructive interference demonstration ---
    interference = logit_step["interference_demo"]
    at_correct = interference["at_correct"]
    at_wrong = interference["at_wrong_example"]

    freq_labels_interf = [f"k={k}" for k in at_correct.keys()]
    fig.add_trace(go.Bar(
        name=f"cos(ωk·0) at c={correct} (all = 1.0)",
        x=freq_labels_interf,
        y=list(at_correct.values()),
        marker_color="green",
        opacity=0.8,
    ), row=4, col=1)
    fig.add_trace(go.Bar(
        name=f"cos(ωk·Δ) at c={(correct+1)%trace['input']['P']} (scattered)",
        x=freq_labels_interf,
        y=list(at_wrong.values()),
        marker_color="red",
        opacity=0.8,
    ), row=4, col=1)

    # --- Panel 8: Probability distribution ---
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    # Show window around correct answer
    fig.add_trace(go.Bar(
        x=x_range,
        y=probs[start:end],
        marker_color=["green" if i == correct else "steelblue" for i in x_range],
        name="P(c|a,b)",
        hovertemplate="c=%{x}: P=%{y:.4f}<extra></extra>",
    ), row=4, col=2)

    fig.update_layout(
        height=1400,
        width=1200,
        title_text=(
            f"Full Circuit Trace: ({a} + {b}) mod {P} = {correct} | "
            f"Predicted: {trace['summary']['predicted']} | "
            f"{'✅ CORRECT' if trace['summary']['is_correct'] else '❌ WRONG'} | "
            f"Confidence: {trace['summary']['confidence']*100:.1f}%"
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05),
    )

    return fig


# =============================================================================
# Visualization: Neuron-Level Detail for a Single Frequency
# =============================================================================

def make_neuron_frequency_detail_figure(trace: dict, frequency: int) -> go.Figure:
    """
    Zoom into a single frequency and show every neuron assigned to it:
    - What each neuron's pre-activation is
    - The ideal cos(ωk(a+b)) value
    - Which neurons fire (post-ReLU)
    - How they contribute to the output
    """
    a = trace["input"]["a"]
    b = trace["input"]["b"]
    P = trace["input"]["P"]
    correct = trace["input"]["correct"]

    mlp_step = trace["steps"][3]
    neurons_for_freq = mlp_step["neurons_by_frequency"].get(frequency, {})

    if not neurons_for_freq or not neurons_for_freq.get("details"):
        fig = go.Figure()
        fig.update_layout(title=f"No neurons assigned to frequency k={frequency}")
        return fig

    details = neurons_for_freq["details"]
    omega_k = 2 * np.pi * frequency / P
    ideal_cos = np.cos(omega_k * (a + b))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Pre-ReLU Activations (k={frequency})",
            f"Post-ReLU Activations (k={frequency})",
            f"Trig Identity Check",
            f"Neuron Contribution Magnitudes",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    neuron_indices = [d["neuron_idx"] for d in details]
    pre_vals = [d["actual_pre_activation"] for d in details]
    post_vals = [d["actual_post_relu"] for d in details]
    trig_products = [d["trig_product"] for d in details]

    # Panel 1: Pre-ReLU
    colors_pre = ["green" if v > 0 else "red" for v in pre_vals]
    fig.add_trace(go.Bar(
        x=[str(i) for i in neuron_indices],
        y=pre_vals,
        marker_color=colors_pre,
        name="Pre-ReLU",
        hovertemplate="Neuron %{x}: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    # Add ideal line
    fig.add_hline(y=ideal_cos, line_dash="dash", line_color="purple",
                  annotation_text=f"cos(ω_{frequency}·{a+b}) = {ideal_cos:.4f}",
                  row=1, col=1)

    # Panel 2: Post-ReLU
    colors_post = ["green" if v > 0 else "lightgray" for v in post_vals]
    fig.add_trace(go.Bar(
        x=[str(i) for i in neuron_indices],
        y=post_vals,
        marker_color=colors_post,
        name="Post-ReLU",
        hovertemplate="Neuron %{x}: %{y:.4f}<extra></extra>",
    ), row=1, col=2)

    # Panel 3: Trig identity verification
    fig.add_trace(go.Scatter(
        x=[str(i) for i in neuron_indices],
        y=trig_products,
        mode="markers",
        marker=dict(size=8, color="blue"),
        name="cos(a)cos(b) - sin(a)sin(b)",
    ), row=2, col=1)
    fig.add_hline(y=ideal_cos, line_dash="dash", line_color="green",
                  annotation_text=f"= cos(ω·(a+b)) = {ideal_cos:.4f}",
                  row=2, col=1)

    # Panel 4: Contribution magnitudes (|post_relu| as proxy)
    contributions = [abs(v) for v in post_vals]
    fig.add_trace(go.Bar(
        x=[str(i) for i in neuron_indices],
        y=contributions,
        marker_color="orange",
        name="|contribution|",
    ), row=2, col=2)

    fig.update_layout(
        height=700,
        title_text=(
            f"Frequency k={frequency} Detail | ω_{frequency} = 2π·{frequency}/{P} = {omega_k:.4f} | "
            f"cos(ω·({a}+{b})) = cos({omega_k*(a+b):.4f}) = {ideal_cos:.4f} | "
            f"{neurons_for_freq['firing']}/{neurons_for_freq['total']} neurons fire"
        ),
        showlegend=False,
    )

    return fig


# =============================================================================
# Visualization: Step-by-Step Equation Substitution (Text-Based)
# =============================================================================

def format_equation_trace_text(trace: dict, max_neurons_shown: int = 5) -> str:
    """
    Generate a full text-based equation trace showing abstract → concrete
    substitution at every step. This is the "show me the math" view.

    Returns a formatted string suitable for display in a Markdown block.
    """
    a = trace["input"]["a"]
    b = trace["input"]["b"]
    P = trace["input"]["P"]
    correct = trace["input"]["correct"]
    key_freqs = trace["weights_info"]["key_frequencies"]

    lines = []
    lines.append(f"# Full Equation Trace: ({a} + {b}) mod {P} = {correct}")
    lines.append(f"")
    lines.append(f"Key frequencies: K = {{{', '.join(str(k) for k in key_freqs)}}}")
    lines.append(f"")

    # ─── STEP 1 ───
    lines.append("=" * 70)
    lines.append("STEP 1: EMBEDDING")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Abstract:")
    lines.append("  x₀ = W_E[a] + W_P[0]    ∈ ℝ^{d_model}")
    lines.append("  x₁ = W_E[b] + W_P[1]    ∈ ℝ^{d_model}")
    lines.append("")
    lines.append(f"Concrete (a={a}, b={b}):")
    lines.append(f"  x₀ = W_E[{a}] + W_P[0]    ||x₀|| = {trace['steps'][0]['values']['x0_norm']:.4f}")
    lines.append(f"  x₁ = W_E[{b}] + W_P[1]    ||x₁|| = {trace['steps'][0]['values']['x1_norm']:.4f}")
    lines.append("")
    lines.append("Fourier decomposition of embeddings at key frequencies:")
    lines.append("")

    fourier = trace["steps"][0]["fourier_decomposition"]
    lines.append(f"  {'k':>4} | {'ω_k':>8} | {'cos(ω_k·a)':>12} | {'sin(ω_k·a)':>12} | "
                 f"{'cos(ω_k·b)':>12} | {'sin(ω_k·b)':>12} | {'cos(ω_k(a+b))':>14}")
    lines.append(f"  {'-'*4}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}")

    for k in key_freqs:
        f = fourier[k]
        lines.append(
            f"  {k:>4} | {f['omega']:>8.4f} | {f['cos_a']:>12.6f} | {f['sin_a']:>12.6f} | "
            f"{f['cos_b']:>12.6f} | {f['sin_b']:>12.6f} | {f['cos_apb']:>14.6f}"
        )
    lines.append("")

    # ─── STEP 2 ───
    lines.append("=" * 70)
    lines.append("STEP 2: ATTENTION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Abstract:")
    lines.append("  For each head h:")
    lines.append("    q^(h) = x₂ · W_Q^(h)")
    lines.append("    k_i^(h) = x_i · W_K^(h)    for i ∈ {0, 1}")
    lines.append("    A_i^(h) = softmax(q · k_i^T / √d_h)")
    lines.append("    head^(h) = Σ_i A_i^(h) · (x_i · W_V^(h))")
    lines.append("  Attn(x) = [head^(0) ‖ ... ‖ head^(H-1)] · W_O")
    lines.append("")
    lines.append(f"Concrete (a={a}, b={b}):")

    attn_data = trace["steps"][1]["concrete_values"]["attention_weights_per_head"]
    for h in sorted(attn_data.keys()):
        w = attn_data[h]
        lines.append(f"  Head {h}: A_to_a = {w['to_a']:.4f}, A_to_b = {w['to_b']:.4f}"
                     f"  {'(≈ uniform)' if abs(w['to_a'] - 0.5) < 0.1 else ''}")
    lines.append("")
    lines.append(f"  ||Attn output|| = {trace['steps'][1]['concrete_values']['combined_norm']:.4f}")
    lines.append("")

    # ─── STEP 3 ───
    lines.append("=" * 70)
    lines.append("STEP 3: RESIDUAL MID")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Abstract:")
    lines.append("  r_mid = x₂ + Attn(x)")
    lines.append("")
    lines.append(f"Concrete:")
    lines.append(f"  ||r_mid|| = {trace['steps'][2]['values']['norm']:.4f}")
    lines.append("")

    # ─── STEP 4 ───
    lines.append("=" * 70)
    lines.append("STEP 4: MLP (TRIG IDENTITY COMPUTATION)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Abstract:")
    lines.append("  z_n = r_mid · W_in[:, n] + b_in[n]")
    lines.append("      ≈ γ_n · [cos(ω_k·a)·cos(ω_k·b) - sin(ω_k·a)·sin(ω_k·b)]")
    lines.append("      = γ_n · cos(ω_k·(a+b))          ← COSINE ADDITION FORMULA")
    lines.append("  m_n = ReLU(z_n)")
    lines.append("  MLP_out = m · W_out + b_out")
    lines.append("")
    lines.append(f"Concrete (a={a}, b={b}):")
    lines.append(f"  Total active neurons: {trace['steps'][3]['total_active_neurons']}")
    lines.append(f"  Key neurons active: {trace['steps'][3]['key_active_neurons']}")
    lines.append("")

    # Show details per frequency
    neurons_by_freq = trace["steps"][3]["neurons_by_frequency"]
    for k in sorted(neurons_by_freq.keys()):
        freq_info = neurons_by_freq[k]
        omega_k = 2 * np.pi * k / P
        ideal = np.cos(omega_k * (a + b))
        lines.append(f"  Frequency k={k}: ω_{k} = {omega_k:.4f}")
        lines.append(f"    Ideal: cos(ω_{k}·({a}+{b})) = cos({omega_k*(a+b):.4f}) = {ideal:.6f}")
        lines.append(f"    Neurons: {freq_info['firing']}/{freq_info['total']} firing")
        lines.append("")

        # Show individual neurons (limited)
        shown = 0
        for d in freq_info["details"]:
            if shown >= max_neurons_shown:
                remaining = len(freq_info["details"]) - shown
                if remaining > 0:
                    lines.append(f"      ... and {remaining} more neurons")
                break

            status = "FIRES" if d["actual_post_relu"] > 0 else "dead "
            lines.append(
                f"      Neuron {d['neuron_idx']:>3}: "
                f"cos({omega_k:.3f}·{a})·cos({omega_k:.3f}·{b}) - "
                f"sin({omega_k:.3f}·{a})·sin({omega_k:.3f}·{b})"
            )
            lines.append(
                f"               = {d['cos_a']:.4f}·{d['cos_b']:.4f} - "
                f"({d['sin_a']:.4f})·({d['sin_b']:.4f})"
            )
            lines.append(
                f"               = {d['trig_product']:.6f}  "
                f"(ideal: {ideal:.6f})  "
                f"actual_pre: {d['actual_pre_activation']:.4f}  "
                f"[{status}]"
            )
            lines.append("")
            shown += 1

    # ─── STEP 5 ───
    lines.append("=" * 70)
    lines.append("STEP 5: FINAL RESIDUAL")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Abstract:")
    lines.append("  r_final = r_mid + MLP(r_mid)")
    lines.append("")
    lines.append(f"Concrete:")
    lines.append(f"  ||r_final|| = {trace['steps'][4]['values']['norm']:.4f}")
    lines.append("")

    # ─── STEP 6 ───
    lines.append("=" * 70)
    lines.append("STEP 6: UNEMBEDDING → LOGITS → PREDICTION")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Abstract:")
    lines.append("  Logit(c) = r_final · W_U[:, c]")
    lines.append(f"           ≈ Σ_{{k∈K}} α_k · cos(2πk(a+b-c)/{P})")
    lines.append("")
    lines.append("Why it works:")
    lines.append(f"  At c = (a+b) mod {P} = {correct}:")
    lines.append(f"    cos(2πk·0/{P}) = cos(0) = 1  for ALL k")
    lines.append(f"    → All frequencies add constructively!")
    lines.append("")
    lines.append(f"  At c ≠ {correct} (e.g., c={(correct+7)%P}):")
    lines.append(f"    cos(2πk·7/{P}) varies per k → destructive interference")
    lines.append("")
    lines.append(f"Concrete:")
    lines.append(f"  Correlation (actual vs ideal formula): "
                 f"{trace['steps'][5]['values']['correlation_actual_vs_ideal']:.4f}")
    lines.append("")

    # Interference demo
    lines.append(f"  Constructive interference at c={correct}:")
    interf = trace["steps"][5]["interference_demo"]["at_correct"]
    for k, val in interf.items():
        lines.append(f"    k={k}: cos(2π·{k}·({a}+{b}-{correct})/{P}) = cos(0) = {val:.6f}")
    lines.append(f"    SUM = {sum(interf.values()):.4f} (all positive!)")
    lines.append("")

    lines.append(f"  Destructive interference at c={(correct+1)%P}:")
    interf_wrong = trace["steps"][5]["interference_demo"]["at_wrong_example"]
    for k, val in interf_wrong.items():
        lines.append(f"    k={k}: cos(2π·{k}·({a}+{b}-{(correct+1)%P})/{P}) = {val:.6f}")
    lines.append(f"    SUM = {sum(interf_wrong.values()):.4f} (cancels out!)")
    lines.append("")

    # Top predictions
    lines.append("  Top 5 predictions:")
    for idx, logit_val, prob in trace["steps"][5]["values"]["top_k"]:
        marker = " ← CORRECT" if idx == correct else ""
        lines.append(f"    c={idx:>3}: logit={logit_val:>8.4f}, P(c|a,b)={prob*100:>6.2f}%{marker}")
    lines.append("")

    # ─── FINAL ───
    lines.append("=" * 70)
    lines.append("RESULT")
    lines.append("=" * 70)
    lines.append("")
    predicted = trace["summary"]["predicted"]
    is_correct = trace["summary"]["is_correct"]
    confidence = trace["summary"]["confidence"]
    lines.append(f"  Input: ({a} + {b}) mod {P}")
    lines.append(f"  Correct answer: {correct}")
    lines.append(f"  Model prediction: {predicted}")
    lines.append(f"  Confidence: {confidence*100:.2f}%")
    lines.append(f"  Status: {'✅ CORRECT' if is_correct else '❌ WRONG'}")

    return "\n".join(lines)


# =============================================================================
# Interactive Comparison: Multiple (a, b) Pairs Side by Side
# =============================================================================

def compare_multiple_inputs(model, pairs: list[tuple[int, int]],
                            key_frequencies: list[int],
                            neuron_assignments: dict) -> go.Figure:
    """
    Run multiple (a, b) pairs and show how the circuit behaves differently
    for each. Useful for seeing patterns (e.g., same sum, different inputs).
    """
    P = model.P
    n_pairs = len(pairs)

    fig = make_subplots(
        rows=3, cols=n_pairs,
        subplot_titles=[
            f"({a}+{b})%{P}={( a+b)%P}" for a, b in pairs
        ] * 3,
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    for col_idx, (a, b) in enumerate(pairs, 1):
        trace = trace_full_computation(model, a, b, key_frequencies, neuron_assignments)
        correct = (a + b) % P

        # Row 1: Fourier components cos(ωk·a) and cos(ωk·b)
        fourier = trace["steps"][0]["fourier_decomposition"]
        freqs = sorted(fourier.keys())
        cos_apb = [fourier[k]["cos_apb"] for k in freqs]

        fig.add_trace(go.Bar(
            x=[f"k={k}" for k in freqs],
            y=cos_apb,
            marker_color=["green" if v > 0 else "red" for v in cos_apb],
            name=f"cos(ωk·{a+b})" if col_idx == 1 else None,
            showlegend=(col_idx == 1),
            hovertemplate=f"({a}+{b}): k=%{{x}}, cos(ωk·{a+b})=%{{y:.4f}}<extra></extra>",
        ), row=1, col=col_idx)

        # Row 2: MLP neuron activations (post-ReLU, key neurons only)
        mlp_step = trace["steps"][3]
        active_details = [d for d in mlp_step["key_neuron_details"]
                          if d["actual_post_relu"] > 0][:20]

        if active_details:
            fig.add_trace(go.Bar(
                x=[str(d["neuron_idx"]) for d in active_details],
                y=[d["actual_post_relu"] for d in active_details],
                marker_color=[f"hsl({(d['frequency']*60)%360}, 70%, 50%)"
                              for d in active_details],
                name=f"Active neurons" if col_idx == 1 else None,
                showlegend=(col_idx == 1),
            ), row=2, col=col_idx)

        # Row 3: Logits around correct answer
        logits = trace["steps"][5]["values"]["logits"]
        window = 10
        start = max(0, correct - window)
        end = min(P, correct + window)
        x_range = list(range(start, end))

        fig.add_trace(go.Bar(
            x=x_range,
            y=logits[start:end],
            marker_color=["green" if i == correct else "steelblue" for i in x_range],
            name=f"Logits" if col_idx == 1 else None,
            showlegend=(col_idx == 1),
        ), row=3, col=col_idx)

    fig.update_layout(
        height=900,
        width=300 * n_pairs + 100,
        title_text="Comparison: Same Circuit, Different Inputs",
        showlegend=True,
    )

    return fig


# =============================================================================
# The Main Interactive Function (for Gradio integration)
# =============================================================================

def run_circuit_equation_viewer(model, a: int, b: int,
                                 key_frequencies: list[int],
                                 neuron_assignments: dict,
                                 selected_frequency: Optional[int] = None) -> dict:
    """
    Main entry point for the interactive circuit equation viewer.
    Call this from the Gradio UI to get all visualizations and text.

    Args:
        model: trained ModularAdditionTransformer
        a, b: input integers
        key_frequencies: discovered key frequencies
        neuron_assignments: dict mapping neuron_idx -> {frequency, ...}
        selected_frequency: if set, also generate per-frequency detail

    Returns dict with:
        - "trace": the full computation trace dict
        - "flow_figure": the multi-panel Plotly figure
        - "equation_text": the full text-based equation trace
        - "freq_detail_figure": per-frequency detail (if selected_frequency given)
    """
    # Run the full trace
    trace = trace_full_computation(model, a, b, key_frequencies, neuron_assignments)

    # Generate the multi-panel figure
    flow_figure = make_equation_flow_figure(trace)

    # Generate the text-based equation trace
    equation_text = format_equation_trace_text(trace)

    # Per-frequency detail (optional)
    freq_detail_figure = None
    if selected_frequency is not None and selected_frequency in key_frequencies:
        freq_detail_figure = make_neuron_frequency_detail_figure(trace, selected_frequency)

    return {
        "trace": trace,
        "flow_figure": flow_figure,
        "equation_text": equation_text,
        "freq_detail_figure": freq_detail_figure,
    }


# =============================================================================
# Gradio Tab Builder (add this to your existing build_gui())
# =============================================================================

def build_equation_viewer_tab(state: dict):
    """
    Build the Gradio tab for the interactive circuit equation viewer.
    Call this inside your existing build_gui() function's gr.Tabs() block.

    Usage in grok.py:
        with gr.TabItem("🔢 Circuit Equations (Live)"):
            build_equation_viewer_tab(state)
    """
    import gradio as gr

    gr.Markdown("### Interactive Circuit Equation Viewer")
    gr.Markdown(
        "Enter any two numbers to see the **full computation** traced through "
        "every layer — with both abstract equations and concrete variable substitution. "
        "See exactly how the transformer uses Fourier components and trig identities "
        "to compute modular addition."
    )

    with gr.Row():
        eq_a = gr.Number(value=7, label="Input a", precision=0)
        eq_b = gr.Number(value=13, label="Input b", precision=0)
        eq_freq_select = gr.Dropdown(
            choices=[], label="Zoom into frequency (optional)",
            interactive=True, value=None,
        )
        eq_run_btn = gr.Button("🔢 Trace Full Computation", variant="primary", size="lg")

    gr.Markdown("---")

    # Output: Summary
    eq_summary_md = gr.Markdown(label="Computation Summary")

    # Output: Multi-panel figure
    eq_flow_plot = gr.Plot(label="Full Circuit Flow (Visual)")

    # Output: Text-based equation trace
    gr.Markdown("#### Full Equation Trace (Text)")
    eq_text_output = gr.Textbox(
        label="Step-by-Step Equations with Variable Substitution",
        lines=40,
        interactive=False,
        show_copy_button=True,
    )

    # Output: Per-frequency detail
    gr.Markdown("#### Per-Frequency Neuron Detail")
    eq_freq_plot = gr.Plot(label="Frequency Detail")

    # Comparison section
    gr.Markdown("---")
    gr.Markdown("#### Compare Multiple Inputs")
    gr.Markdown(
        "Enter multiple pairs to see how the same circuit handles different inputs. "
        "Format: `a1,b1; a2,b2; a3,b3`"
    )
    eq_compare_input = gr.Textbox(
        label="Pairs to compare",
        placeholder="7,13; 50,63; 100,13; 56,57",
        value="7,13; 50,63; 100,13; 56,57",
    )
    eq_compare_btn = gr.Button("Compare Inputs", variant="secondary")
    eq_compare_plot = gr.Plot(label="Multi-Input Comparison")

    def run_equation_viewer(a, b, freq_select):
        """Run the full equation viewer."""
        if state.get("model") is None:
            return (
                "⚠️ No model loaded! Train or load a model first.",
                go.Figure(),
                "No model available.",
                go.Figure(),
            )

        if state.get("circuit") is None:
            return (
                "⚠️ No circuit discovered! Run Fourier Discovery first.",
                go.Figure(),
                "Run Fourier Discovery first.",
                go.Figure(),
            )

        model = state["model"]
        circuit = state["circuit"]
        P = model.P
        a_int = int(a) % P
        b_int = int(b) % P

        key_frequencies = circuit.key_frequencies
        neuron_assignments = circuit.neuron_frequency_assignments

        # Parse selected frequency
        selected_freq = None
        if freq_select and freq_select != "None":
            try:
                selected_freq = int(freq_select.split("=")[1]) if "=" in freq_select else int(freq_select)
            except (ValueError, IndexError):
                selected_freq = None

        # Run the viewer
        result = run_circuit_equation_viewer(
            model, a_int, b_int,
            key_frequencies, neuron_assignments,
            selected_frequency=selected_freq,
        )

        trace = result["trace"]
        correct = trace["input"]["correct"]
        predicted = trace["summary"]["predicted"]
        confidence = trace["summary"]["confidence"]
        is_correct = trace["summary"]["is_correct"]

        summary_md = (
            f"## ({a_int} + {b_int}) mod {P} = **{correct}**\n\n"
            f"**Prediction:** {predicted} | "
            f"**Confidence:** {confidence*100:.1f}% | "
            f"**Status:** {'✅ Correct' if is_correct else '❌ Wrong'}\n\n"
            f"**Key frequencies:** {key_frequencies}\n\n"
            f"**Correlation (actual vs formula):** "
            f"{trace['steps'][5]['values']['correlation_actual_vs_ideal']:.4f}\n\n"
            f"**Active key neurons:** {trace['steps'][3]['key_active_neurons']} / "
            f"{len(neuron_assignments)}"
        )

        freq_fig = result["freq_detail_figure"] if result["freq_detail_figure"] else go.Figure()

        return (
            summary_md,
            result["flow_figure"],
            result["equation_text"],
            freq_fig,
        )

    def run_comparison(pairs_str):
        """Run comparison of multiple inputs."""
        if state.get("model") is None:
            return go.Figure().update_layout(title="⚠ No model loaded!")

        if state.get("circuit") is None:
            return go.Figure().update_layout(title="⚠ No circuit discovered!")

        model = state["model"]
        circuit = state["circuit"]
        P = model.P
        key_frequencies = circuit.key_frequencies
        neuron_assignments = circuit.neuron_frequency_assignments

        # Parse pairs
        pairs = []
        try:
            for pair_str in pairs_str.strip().split(";"):
                pair_str = pair_str.strip()
                if not pair_str:
                    continue
                parts = pair_str.split(",")
                a_val = int(parts[0].strip()) % P
                b_val = int(parts[1].strip()) % P
                pairs.append((a_val, b_val))
        except (ValueError, IndexError):
            return go.Figure().update_layout(
                title="❌ Invalid format. Use: a1,b1; a2,b2; a3,b3"
            )

        if not pairs:
            return go.Figure().update_layout(title="No valid pairs found.")

        # Limit to 6 pairs for readability
        pairs = pairs[:6]

        return compare_multiple_inputs(model, pairs, key_frequencies, neuron_assignments)

    def update_freq_dropdown():
        """Update the frequency dropdown when circuit is available."""
        if state.get("circuit") is None:
            return gr.Dropdown(choices=[], value=None)
        freqs = state["circuit"].key_frequencies
        choices = [f"k={k}" for k in freqs]
        return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

    # Wire up events
    eq_run_btn.click(
        fn=run_equation_viewer,
        inputs=[eq_a, eq_b, eq_freq_select],
        outputs=[eq_summary_md, eq_flow_plot, eq_text_output, eq_freq_plot],
    )

    eq_compare_btn.click(
        fn=run_comparison,
        inputs=[eq_compare_input],
        outputs=[eq_compare_plot],
    )

    # Update frequency dropdown when tab is opened
    eq_run_btn.click(
        fn=update_freq_dropdown,
        inputs=[],
        outputs=[eq_freq_select],
    )


# =============================================================================
# Rewritten: model.forward_with_hooks() — ensure all needed keys are captured
# =============================================================================

def forward_with_hooks_extended(model, a_tensor, b_tensor):
    """
    Extended forward pass that captures ALL intermediate activations needed
    by the circuit equation viewer. This should REPLACE or EXTEND the existing
    model.forward_with_hooks() method.

    Returns:
        logits: (batch, P) tensor
        activations: dict with keys:
            - "tok_embed": (batch, 3, d_model)
            - "pos_embed": (batch, 3, d_model)
            - "embed": (batch, 3, d_model)  — combined tok + pos
            - "attn_weights": (batch, n_heads, 1, 2) — attention from pos 2 to pos 0,1
            - "attn_head_0" ... "attn_head_{H-1}": (batch, 1, d_head)
            - "attn_out": (batch, 1, d_model) — after W_O projection
            - "residual_mid": (batch, 1, d_model) — x + attn_out
            - "mlp_pre": (batch, 1, d_mlp) — before ReLU
            - "mlp_hidden": (batch, 1, d_mlp) — after ReLU
            - "mlp_out": (batch, 1, d_model) — after W_out projection
            - "residual_final": (batch, 1, d_model) — residual_mid + mlp_out
            - "logits": (batch, P)
    """
    P = model.P
    d_model = model.d_model
    n_heads = model.n_heads
    d_head = d_model // n_heads
    batch = a_tensor.shape[0]

    activations = {}

    # === Embedding ===
    # Token embeddings
    eq_token_idx = P  # The "=" token is at index P
    tok_a = model.embed.weight[a_tensor]          # (batch, d_model)
    tok_b = model.embed.weight[b_tensor]          # (batch, d_model)
    tok_eq = model.embed.weight[eq_token_idx].unsqueeze(0).expand(batch, -1)  # (batch, d_model)

    tok_embed = torch.stack([tok_a, tok_b, tok_eq], dim=1)  # (batch, 3, d_model)
    activations["tok_embed"] = tok_embed.detach()

    # Positional embeddings
    pos_embed = model.pos_embed.weight.unsqueeze(0).expand(batch, -1, -1)  # (batch, 3, d_model)
    activations["pos_embed"] = pos_embed.detach()

    # Combined
    x = tok_embed + pos_embed  # (batch, 3, d_model)
    activations["embed"] = x.detach()

    # === Attention ===
    # We only care about position 2 (the "=" position) attending to positions 0 and 1
    x_eq = x[:, 2:3, :]   # (batch, 1, d_model) — query source
    x_ab = x[:, :2, :]    # (batch, 2, d_model) — key/value sources

    # Compute Q, K, V for all heads
    # model.attn.W_Q: (n_heads, d_model, d_head)
    # model.attn.W_K: (n_heads, d_model, d_head)
    # model.attn.W_V: (n_heads, d_model, d_head)

    # Q from position 2: (batch, 1, d_model) @ (n_heads, d_model, d_head) -> (batch, n_heads, 1, d_head)
    Q = torch.einsum("bsd,hde->bhse", x_eq, model.attn.W_Q)  # (batch, n_heads, 1, d_head)

    # K from positions 0,1: (batch, 2, d_model) @ (n_heads, d_model, d_head) -> (batch, n_heads, 2, d_head)
    K = torch.einsum("bsd,hde->bhse", x_ab, model.attn.W_K)  # (batch, n_heads, 2, d_head)

    # V from positions 0,1
    V = torch.einsum("bsd,hde->bhse", x_ab, model.attn.W_V)  # (batch, n_heads, 2, d_head)

    # Attention scores: Q @ K^T / sqrt(d_head)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (d_head ** 0.5)  # (batch, n_heads, 1, 2)

    # Attention weights (softmax over the 2 key positions)
    attn_weights = torch.softmax(scores, dim=-1)  # (batch, n_heads, 1, 2)
    activations["attn_weights"] = attn_weights.detach()

    # Weighted sum of values: (batch, n_heads, 1, 2) @ (batch, n_heads, 2, d_head) -> (batch, n_heads, 1, d_head)
    head_outputs = torch.einsum("bhqk,bhkd->bhqd", attn_weights, V)  # (batch, n_heads, 1, d_head)

    # Store per-head outputs
    for h in range(n_heads):
        activations[f"attn_head_{h}"] = head_outputs[:, h, :, :].detach()  # (batch, 1, d_head)

    # Apply W_O: (batch, n_heads, 1, d_head) -> (batch, 1, d_model)
    # model.attn.W_O: (n_heads, d_head, d_model)
    attn_out = torch.einsum("bhsd,hde->bse", head_outputs, model.attn.W_O)  # (batch, 1, d_model)
    activations["attn_out"] = attn_out.detach()

    # === Residual Mid ===
    residual_mid = x_eq + attn_out  # (batch, 1, d_model)
    activations["residual_mid"] = residual_mid.detach()

    # === MLP ===
    # model.mlp.W_in: Linear(d_model, d_mlp) — weight is (d_mlp, d_model), bias is (d_mlp,)
    # model.mlp.W_out: Linear(d_mlp, d_model) — weight is (d_model, d_mlp), bias is (d_model,)

    mlp_input = residual_mid.squeeze(1)  # (batch, d_model)

    # Pre-activation
    mlp_pre = model.mlp.W_in(mlp_input)  # (batch, d_mlp)
    activations["mlp_pre"] = mlp_pre.unsqueeze(1).detach()  # (batch, 1, d_mlp)

    # ReLU
    mlp_hidden = torch.relu(mlp_pre)  # (batch, d_mlp)
    activations["mlp_hidden"] = mlp_hidden.unsqueeze(1).detach()  # (batch, 1, d_mlp)

    # Output projection
    mlp_out = model.mlp.W_out(mlp_hidden)  # (batch, d_model)
    activations["mlp_out"] = mlp_out.unsqueeze(1).detach()  # (batch, 1, d_model)

    # === Final Residual ===
    residual_final = residual_mid.squeeze(1) + mlp_out  # (batch, d_model)
    activations["residual_final"] = residual_final.unsqueeze(1).detach()  # (batch, 1, d_model)

    # === Unembedding ===
    # model.unembed: Linear(d_model, P) or Embedding used as linear
    logits = model.unembed(residual_final)  # (batch, P)
    activations["logits"] = logits.detach()

    return logits, activations


# =============================================================================
# Integration: Patch into grok.py's build_gui()
# =============================================================================

def integrate_equation_viewer_into_gui(build_gui_func):
    """
    Decorator/wrapper that adds the Circuit Equations tab to the existing GUI.

    Usage in grok.py — replace the last section:

        # At the end of build_gui(), inside the gr.Tabs() block, add:
        with gr.TabItem("🔢 Circuit Equations (Live)"):
            build_equation_viewer_tab(state)

    Or use this function to wrap the entire build_gui:

        demo = integrate_equation_viewer_into_gui(build_gui)()
    """
    import functools

    @functools.wraps(build_gui_func)
    def wrapped():
        # This is a placeholder showing WHERE to insert the tab.
        # In practice, you add the tab directly inside build_gui().
        return build_gui_func()

    return wrapped


# =============================================================================
# Standalone Test / Demo
# =============================================================================

def demo_equation_viewer():
    """
    Standalone demo: loads a model from saved_runs (if available) and
    runs the equation viewer on a sample input.
    """
    print("=" * 70)
    print("Circuit Equation Viewer — Standalone Demo")
    print("=" * 70)

    # Try to load a saved model
    runs = list_saved_runs() if 'list_saved_runs' in dir() else []
    grokked_runs = [r for r in runs if r.get("grokked")]

    if not grokked_runs:
        print("\n⚠ No grokked model found in saved_runs/.")
        print("  Train a model first using the GUI, then run this demo.")
        print("  Or run: uv run grok.py")
        return

    # Load the first grokked run
    run_id = grokked_runs[0]["run_id"]
    print(f"\nLoading run: {run_id}")

    data = load_run(run_id)
    model = data["model"]
    P = data["config"]["P"]

    print(f"Model loaded: P={P}, d_model={data['config']['d_model']}")

    # Run circuit discovery
    print("\nRunning Fourier circuit discovery...")
    from circuit_discoverer import CircuitDiscoverer
    discoverer = CircuitDiscoverer(model)
    circuit = discoverer.full_discovery(progress_cb=lambda msg: None)

    print(f"Key frequencies: {circuit.key_frequencies}")
    print(f"Neurons assigned: {len(circuit.neuron_frequency_assignments)}")
    print(f"FVE: {circuit.fve_logits:.4f}")
    print(f"Verification accuracy: {circuit.verification_accuracy*100:.2f}%")

    # Run the equation viewer on a sample input
    a, b = 7, 13
    print(f"\n{'=' * 70}")
    print(f"Tracing computation for ({a} + {b}) mod {P} = {(a+b)%P}")
    print(f"{'=' * 70}")

    trace = trace_full_computation(
        model, a, b,
        circuit.key_frequencies,
        circuit.neuron_frequency_assignments,
    )

    # Print the full text trace
    text = format_equation_trace_text(trace)
    print(text)

    # Generate and save the flow figure
    fig = make_equation_flow_figure(trace)
    fig.write_html("circuit_equation_trace.html")
    print(f"\n✅ Flow figure saved to: circuit_equation_trace.html")

    # Generate per-frequency detail for the first key frequency
    if circuit.key_frequencies:
        freq = circuit.key_frequencies[0]
        freq_fig = make_neuron_frequency_detail_figure(trace, freq)
        freq_fig.write_html(f"circuit_freq_{freq}_detail.html")
        print(f"✅ Frequency detail saved to: circuit_freq_{freq}_detail.html")

    # Run comparison
    pairs = [(7, 13), (50, 63), (100, 13), (56, 57)]
    comp_fig = compare_multiple_inputs(
        model, pairs,
        circuit.key_frequencies,
        circuit.neuron_frequency_assignments,
    )
    comp_fig.write_html("circuit_comparison.html")
    print(f"✅ Comparison figure saved to: circuit_comparison.html")

    print(f"\n{'=' * 70}")
    print("Demo complete! Open the HTML files in a browser to explore.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    demo_equation_viewer()
