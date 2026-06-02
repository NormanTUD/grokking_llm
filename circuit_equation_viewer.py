"""
Circuit Equation Viewer — Shows the FULL computation of (a + b) mod P
with Temml-rendered equations AND plots.

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
    modular addition circuit.

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
        W_E = model.embed.weight[:P].cpu().numpy()
        W_P = model.pos_embed.weight.cpu().numpy()
        W_U = model.unembed.weight.cpu().numpy()

        W_Q_full = model.W_Q.weight.cpu().numpy()
        W_K_full = model.W_K.weight.cpu().numpy()
        W_V_full = model.W_V.weight.cpu().numpy()
        W_O_full = model.W_O.weight.cpu().numpy()

        W_Q_T = W_Q_full.T
        W_K_T = W_K_full.T
        W_V_T = W_V_full.T
        W_O_T = W_O_full.T

        W_Q = W_Q_T.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        W_K = W_K_T.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        W_V = W_V_T.reshape(d_model, n_heads, d_head).transpose(1, 0, 2)
        W_O = W_O_T.reshape(n_heads, d_head, d_model)

        W_in = model.mlp_in.weight.cpu().numpy()
        b_in = model.mlp_in.bias.cpu().numpy()
        W_out = model.mlp_out.weight.cpu().numpy()
        b_out = model.mlp_out.bias.cpu().numpy()

    key_neuron_indices = sorted(int(idx) for idx in neuron_assignments.keys())

    fourier_components = {}
    for k in key_frequencies:
        omega_k = 2 * np.pi * k / P
        cos_vals = np.cos(omega_k * np.arange(P))
        sin_vals = np.sin(omega_k * np.arange(P))
        fourier_components[k] = {"cos": cos_vals, "sin": sin_vals, "omega": omega_k}

    return {
        "W_E": W_E, "W_P": W_P, "W_Q": W_Q, "W_K": W_K, "W_V": W_V,
        "W_O": W_O, "W_in": W_in, "b_in": b_in, "W_out": W_out, "b_out": b_out,
        "W_U": W_U, "P": P, "d_model": d_model, "n_heads": n_heads,
        "d_head": d_head, "d_mlp": W_in.shape[0],
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
    that participates in the circuit.
    """
    P = model.P
    d_model = model.d_model
    n_heads = model.n_heads
    d_head = d_model // n_heads
    correct = (a + b) % P

    weights = extract_relevant_circuit_weights(model, key_frequencies, neuron_assignments)

    a_tensor = torch.tensor([a])
    b_tensor = torch.tensor([b])

    with torch.no_grad():
        logits_full, activations = model.forward_with_hooks(a_tensor, b_tensor)

    embed_actual = activations["embed"][0].cpu().numpy()
    x0_actual = embed_actual[0]
    x1_actual = embed_actual[1]
    x2_actual = embed_actual[2]

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
        "fourier_decomposition": fourier_embed,
        "values": {
            "x0": x0_actual, "x1": x1_actual, "x2": x2_actual,
            "x0_norm": float(np.linalg.norm(x0_actual)),
            "x1_norm": float(np.linalg.norm(x1_actual)),
        },
    }

    # STEP 2: ATTENTION
    attn_weights_actual = activations["attn_weights"][0].cpu().numpy()

    attn_head_outputs = []
    for h in range(n_heads):
        head_out = activations[f"attn_head_{h}"][0, 0].cpu().numpy()
        attn_head_outputs.append(head_out)

    attn_combined = activations["attn_out"][0, 0].cpu().numpy()

    step2 = {
        "name": "Attention",
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
    }

    # STEP 3: RESIDUAL MID
    residual_mid = activations["residual_mid"][0, 0].cpu().numpy()

    step3_residual = {
        "name": "Residual Stream (Mid)",
        "values": {
            "residual_mid": residual_mid,
            "norm": float(np.linalg.norm(residual_mid)),
        },
    }

    # STEP 4: MLP
    mlp_pre = activations["mlp_pre"][0, 0].cpu().numpy()
    mlp_hidden = activations["mlp_hidden"][0, 0].cpu().numpy()
    mlp_out = activations["mlp_out"][0, 0].cpu().numpy()

    key_neuron_details = []
    for neuron_idx_str, info in neuron_assignments.items():
        neuron_idx = int(neuron_idx_str)
        freq = info["frequency"]
        omega_k = 2 * np.pi * freq / P

        cos_a = np.cos(omega_k * a)
        sin_a = np.sin(omega_k * a)
        cos_b = np.cos(omega_k * b)
        sin_b = np.sin(omega_k * b)

        ideal_cos_apb = np.cos(omega_k * (a + b))
        trig_product = cos_a * cos_b - sin_a * sin_b

        actual_pre = float(mlp_pre[neuron_idx])
        actual_post = float(mlp_hidden[neuron_idx])

        key_neuron_details.append({
            "neuron_idx": neuron_idx,
            "frequency": freq,
            "omega_k": omega_k,
            "cos_a": cos_a, "sin_a": sin_a,
            "cos_b": cos_b, "sin_b": sin_b,
            "ideal_cos_apb": ideal_cos_apb,
            "trig_product": trig_product,
            "actual_pre_activation": actual_pre,
            "actual_post_relu": actual_post,
        })

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
        "neurons_by_frequency": neurons_by_freq,
        "key_neuron_details": key_neuron_details,
        "total_active_neurons": int((mlp_hidden > 0).sum()),
        "key_active_neurons": sum(1 for d in key_neuron_details if d["actual_post_relu"] > 0),
        "mlp_output_norm": float(np.linalg.norm(mlp_out)),
    }

    # STEP 5: FINAL RESIDUAL
    residual_final = activations["residual_final"][0, 0].cpu().numpy()

    step5_residual = {
        "name": "Final Residual Stream",
        "values": {
            "residual_final": residual_final,
            "norm": float(np.linalg.norm(residual_final)),
        },
    }

    # STEP 6: UNEMBEDDING
    logits = activations["logits"][0].cpu().numpy()

    ideal_logits = np.zeros(P)
    for k in key_frequencies:
        omega_k = 2 * np.pi * k / P
        for c in range(P):
            ideal_logits[c] += np.cos(omega_k * (a + b - c))

    if np.std(ideal_logits) > 0:
        scale = np.std(logits) / np.std(ideal_logits)
        ideal_logits_scaled = ideal_logits * scale
    else:
        ideal_logits_scaled = ideal_logits

    correlation = float(np.corrcoef(logits, ideal_logits)[0, 1]) if np.std(ideal_logits) > 0 else 0.0

    top_k = 5
    top_indices = np.argsort(logits)[-top_k:][::-1]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()

    step6_logits = {
        "name": "Unembedding (Logits)",
        "values": {
            "logits": logits,
            "ideal_logits": ideal_logits,
            "ideal_logits_scaled": ideal_logits_scaled,
            "correlation_actual_vs_ideal": correlation,
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

    return {
        "input": {"a": a, "b": b, "P": P, "correct": correct},
        "steps": [step1, step2, step3_residual, step4_mlp, step5_residual, step6_logits],
        "summary": {
            "predicted": int(logits.argmax()),
            "correct": correct,
            "is_correct": bool(logits.argmax() == correct),
            "confidence": float(probs[correct]),
            "correlation_with_formula": correlation,
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
# TEMML EQUATION RENDERING — The main new feature
# =============================================================================

def generate_temml_equation_html(trace: dict, show_abstract: bool = True,
                                  show_concrete: bool = True,
                                  max_neurons_shown: int = 5) -> str:
    """
    Generate HTML with Temml-rendered equations showing the complete
    circuit computation. Shows both abstract form and concrete variable
    substitution.
    """
    a = trace["input"]["a"]
    b = trace["input"]["b"]
    P = trace["input"]["P"]
    correct = trace["input"]["correct"]
    predicted = trace["summary"]["predicted"]
    is_correct = trace["summary"]["is_correct"]
    confidence = trace["summary"]["confidence"]
    key_freqs = trace["weights_info"]["key_frequencies"]
    d_model = trace["weights_info"]["d_model"]
    n_heads = trace["weights_info"]["n_heads"]
    d_mlp = trace["weights_info"]["d_mlp"]

    sections = []

    # ─── HEADER ───
    header_latex = (
        rf"\text{{Circuit Equations for }}({a} + {b}) \bmod {P} = {correct}"
        rf"\quad \text{{Predicted: }}{predicted}"
        rf"\quad {'\\color{{green}}\\checkmark' if is_correct else '\\color{{red}}\\times'}"
    )

    # ─── ARCHITECTURE ───
    arch_latex = (
        rf"d_{{\text{{model}}}} = {d_model},\;"
        rf"n_{{\text{{heads}}}} = {n_heads},\;"
        rf"d_{{\text{{head}}}} = {d_model // n_heads},\;"
        rf"d_{{\text{{mlp}}}} = {d_mlp},\;"
        rf"P = {P}"
    )
    freq_str = ", ".join(str(k) for k in key_freqs)
    freq_latex = rf"\mathcal{{K}} = \{{{freq_str}\}}"
    sections.append(("Architecture & Key Frequencies", [arch_latex, freq_latex]))

    # ─── STEP 1: EMBEDDING ───
    embed_eqs = []
    if show_abstract:
        embed_eqs.append(
            r"\mathbf{x}_0 = W_E[a] + W_P[0] \in \mathbb{R}^{" + str(d_model) + r"}"
        )
        embed_eqs.append(
            r"\mathbf{x}_1 = W_E[b] + W_P[1] \in \mathbb{R}^{" + str(d_model) + r"}"
        )
        embed_eqs.append(
            r"\mathbf{x}_2 = W_E[\text{=}] + W_P[2] \in \mathbb{R}^{" + str(d_model) + r"}"
        )
        embed_eqs.append(
            r"W_E[t] \approx \sum_{k \in \mathcal{K}} "
            r"\left[\alpha_k \cos\!\left(\frac{2\pi k t}{" + str(P) + r"}\right) \mathbf{u}_k^{(\cos)}"
            r"+ \beta_k \sin\!\left(\frac{2\pi k t}{" + str(P) + r"}\right) \mathbf{u}_k^{(\sin)}\right]"
        )

    if show_concrete:
        x0_norm = trace["steps"][0]["values"]["x0_norm"]
        x1_norm = trace["steps"][0]["values"]["x1_norm"]
        embed_eqs.append(
            rf"\mathbf{{x}}_0 = W_E[{a}] + W_P[0],\quad \|\mathbf{{x}}_0\| = {x0_norm:.4f}"
        )
        embed_eqs.append(
            rf"\mathbf{{x}}_1 = W_E[{b}] + W_P[1],\quad \|\mathbf{{x}}_1\| = {x1_norm:.4f}"
        )

        fourier = trace["steps"][0]["fourier_decomposition"]
        for k in key_freqs:
            f = fourier[k]
            embed_eqs.append(
                rf"k={k}:\; \omega_{{{k}}} = \frac{{2\pi \cdot {k}}}{{{P}}} = {f['omega']:.4f}"
                rf",\; \cos(\omega_{{{k}}} \cdot {a}) = {f['cos_a']:.4f}"
                rf",\; \sin(\omega_{{{k}}} \cdot {a}) = {f['sin_a']:.4f}"
                rf",\; \cos(\omega_{{{k}}} \cdot {b}) = {f['cos_b']:.4f}"
                rf",\; \sin(\omega_{{{k}}} \cdot {b}) = {f['sin_b']:.4f}"
            )

    sections.append(("Step 1: Embedding", embed_eqs))

    # ─── STEP 2: ATTENTION ───
    attn_eqs = []
    if show_abstract:
        attn_eqs.append(
            r"\mathbf{q}^{(h)} = \mathbf{x}_2 \cdot W_Q^{(h)} \in \mathbb{R}^{d_h}"
        )
        attn_eqs.append(
            r"\mathbf{k}_i^{(h)} = \mathbf{x}_i \cdot W_K^{(h)},\quad "
            r"\mathbf{v}_i^{(h)} = \mathbf{x}_i \cdot W_V^{(h)}"
        )
        attn_eqs.append(
            r"A_i^{(h)} = \text{softmax}\!\left(\frac{\mathbf{q}^{(h)} \cdot "
            r"\mathbf{k}_i^{(h)\top}}{\sqrt{d_h}}\right)"
        )
        attn_eqs.append(
            r"\text{Attn}(\mathbf{x}) = \sum_{h=0}^{" + str(n_heads-1) + r"}"
            r"\left(A_0^{(h)} \mathbf{v}_0^{(h)} + A_1^{(h)} \mathbf{v}_1^{(h)}\right) W_O^{(h)}"
        )

    if show_concrete:
        attn_data = trace["steps"][1]["concrete_values"]["attention_weights_per_head"]
        for h in sorted(attn_data.keys()):
            w = attn_data[h]
            uniform_note = r"\;\approx\;\text{uniform}" if abs(w['to_a'] - 0.5) < 0.1 else ""
            attn_eqs.append(
                rf"\text{{Head {h}:}}\; A_0^{{({h})}} = {w['to_a']:.4f},\; "
                rf"A_1^{{({h})}} = {w['to_b']:.4f}{uniform_note}"
            )
        combined_norm = trace["steps"][1]["concrete_values"]["combined_norm"]
        attn_eqs.append(
            rf"\|\text{{Attn}}(\mathbf{{x}})\| = {combined_norm:.4f}"
        )

    sections.append(("Step 2: Attention", attn_eqs))

    # ─── STEP 3: RESIDUAL MID ───
    res_mid_eqs = []
    if show_abstract:
        res_mid_eqs.append(
            r"\mathbf{r}_{\text{mid}} = \mathbf{x}_2 + \text{Attn}(\mathbf{x})"
        )
    if show_concrete:
        norm = trace["steps"][2]["values"]["norm"]
        res_mid_eqs.append(
            rf"\|\mathbf{{r}}_{{\text{{mid}}}}\| = {norm:.4f}"
        )
    sections.append(("Step 3: Residual Stream (Mid)", res_mid_eqs))

    # ─── STEP 4: MLP ───
    mlp_eqs = []
    if show_abstract:
        mlp_eqs.append(
            r"z_n = \mathbf{r}_{\text{mid}} \cdot (W_{\text{in}})_{:,n} + (b_{\text{in}})_n"
        )
        mlp_eqs.append(
            r"\approx \gamma_n \left[\cos(\omega_k a)\cos(\omega_k b) "
            r"- \sin(\omega_k a)\sin(\omega_k b)\right]"
        )
        mlp_eqs.append(
            r"= \gamma_n \cos\!\bigl(\omega_k(a+b)\bigr)"
            r"\quad\leftarrow\;\textbf{Cosine Addition Formula}"
        )
        mlp_eqs.append(r"m_n = \text{ReLU}(z_n) = \max(0, z_n)")
        mlp_eqs.append(
            r"\text{MLP}(\mathbf{r}_{\text{mid}}) = \mathbf{m} \cdot W_{\text{out}} + \mathbf{b}_{\text{out}}"
        )

    if show_concrete:
        mlp_step = trace["steps"][3]
        mlp_eqs.append(
            rf"\text{{Active: }} {mlp_step['key_active_neurons']}"
            rf"\text{{ of }} {trace['weights_info']['n_key_neurons']}"
            rf"\text{{ key neurons fire}}"
        )

        neurons_by_freq = mlp_step["neurons_by_frequency"]
        for k in sorted(neurons_by_freq.keys()):
            freq_info = neurons_by_freq[k]
            omega_k = 2 * np.pi * k / P
            ideal = np.cos(omega_k * (a + b))

            mlp_eqs.append(
                rf"\boxed{{k={k}}}:\; \omega_{{{k}}} = {omega_k:.4f},\;"
                rf"\cos(\omega_{{{k}}} \cdot {a+b}) = {ideal:.6f},\;"
                rf"\text{{{freq_info['firing']}/{freq_info['total']} fire}}"
            )

            shown = 0
            for d in freq_info["details"]:
                if shown >= max_neurons_shown:
                    remaining = len(freq_info["details"]) - shown
                    if remaining > 0:
                        mlp_eqs.append(rf"\quad\vdots\quad\text{{({remaining} more)}}")
                    break

                fires = (r"\color{green}\checkmark" if d["actual_post_relu"] > 0
                         else r"\color{gray}\times")

                mlp_eqs.append(
                    rf"\quad n_{{{d['neuron_idx']}}}:\;"
                    rf"\underbrace{{\cos({omega_k:.3f}\!\cdot\!{a})}}_{{ {d['cos_a']:.4f} }}"
                    rf"\!\cdot\!\underbrace{{\cos({omega_k:.3f}\!\cdot\!{b})}}_{{ {d['cos_b']:.4f} }}"
                    rf"\;-\;\underbrace{{\sin({omega_k:.3f}\!\cdot\!{a})}}_{{ {d['sin_a']:.4f} }}"
                    rf"\!\cdot\!\underbrace{{\sin({omega_k:.3f}\!\cdot\!{b})}}_{{ {d['sin_b']:.4f} }}"
                    rf"\;=\;{d['trig_product']:.4f}"
                    rf"\quad z={d['actual_pre_activation']:.3f}\;{fires}"
                )
                shown += 1

    sections.append(("Step 4: MLP — Trig Identity", mlp_eqs))

    # ─── STEP 5: FINAL RESIDUAL ───
    res_final_eqs = []
    if show_abstract:
        res_final_eqs.append(
            r"\mathbf{r}_{\text{final}} = \mathbf{r}_{\text{mid}} + \text{MLP}(\mathbf{r}_{\text{mid}})"
        )
    if show_concrete:
        norm = trace["steps"][4]["values"]["norm"]
        res_final_eqs.append(rf"\|\mathbf{{r}}_{{\text{{final}}}}\| = {norm:.4f}")
    sections.append(("Step 5: Final Residual", res_final_eqs))

    # ─── STEP 6: UNEMBEDDING ───
    logit_eqs = []
    if show_abstract:
        logit_eqs.append(
            r"\text{Logit}(c) = \mathbf{r}_{\text{final}} \cdot (W_U)_{c,:}"
        )
        logit_eqs.append(
            r"\approx \sum_{k \in \mathcal{K}} \alpha_k "
            r"\cos\!\left(\frac{2\pi k(a+b-c)}{" + str(P) + r"}\right)"
        )
        logit_eqs.append(
            r"\hat{c} = \arg\max_c \;\text{Logit}(c) = (a+b) \bmod " + str(P)
        )

    if show_concrete:
        logit_step = trace["steps"][5]
        corr = logit_step["values"]["correlation_actual_vs_ideal"]
        logit_val = logit_step["values"]["logit_at_correct"]
        prob_val = logit_step["values"]["prob_at_correct"]

        # Constructive interference
        logit_eqs.append(
            rf"\text{{At }}\;c = {correct}:\quad"
            + "+".join(
                rf"\cos\!\left(\frac{{2\pi\!\cdot\!{k}\!\cdot\!0}}{{{P}}}\right)"
                for k in key_freqs[:5]
            )
            + (r"+\cdots" if len(key_freqs) > 5 else "")
            + rf" = {len(key_freqs)} \times 1"
            + r"\;\;\color{green}\text{(constructive!)}"
        )

        # Destructive interference
        wrong_c = (correct + 1) % P
        interf_wrong = logit_step["interference_demo"]["at_wrong_example"]
        wrong_sum = sum(interf_wrong.values())
        wrong_terms = list(interf_wrong.items())[:5]
        logit_eqs.append(
            rf"\text{{At }}\;c = {wrong_c}:\quad"
            + "+".join(
                rf"({v:.3f})" for k, v in wrong_terms
            )
            + (r"+\cdots" if len(interf_wrong) > 5 else "")
            + rf" = {wrong_sum:.3f}"
            + r"\;\;\color{red}\text{(destructive!)}"
        )

        logit_eqs.append(
            rf"\text{{Logit}}({correct}) = {logit_val:.4f},\quad"
            rf"P(c\!=\!{correct}\mid {a},{b}) = {prob_val:.4f},\quad"
            rf"\rho_{{\text{{actual vs ideal}}}} = {corr:.4f}"
        )

    sections.append(("Step 6: Unembedding → Prediction", logit_eqs))

    # ─── RESULT ───
    result_eqs = [
        rf"\boxed{{({a} + {b}) \bmod {P} = {correct}}}",
        rf"\hat{{c}} = {predicted}\quad"
        + (r"\color{green}\checkmark\;\text{CORRECT}" if is_correct
           else rf"\color{{red}}\times\;\text{{WRONG (expected {correct})}}"),
        rf"\text{{Confidence: }} {confidence*100:.1f}\%",
    ]
    sections.append(("Result", result_eqs))

    # ─── BUILD HTML ───
    return _build_temml_html(header_latex, sections)


def _build_temml_html(header_latex: str, sections: list[tuple[str, list[str]]]) -> str:
    """
    Build final HTML with Temml CDN rendering all LaTeX equations.
    """
    def escape_for_html(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    equation_blocks = []

    # Header
    equation_blocks.append(f'''
        <div class="eq-header">
            <span class="eq-math" data-tex="{escape_for_html(header_latex)}"></span>
        </div>
    ''')

    # Sections
    for section_title, equations in sections:
        eq_html_items = []
        for eq in equations:
            eq_html_items.append(
                f'<div class="eq-line">'
                f'<span class="eq-math" data-tex="{escape_for_html(eq)}"></span>'
                f'</div>'
            )

        equation_blocks.append(f'''
            <div class="eq-section">
                <h3 class="eq-section-title">{section_title}</h3>
                {"".join(eq_html_items)}
            </div>
        ''')

    # Generate a unique ID to avoid conflicts if multiple instances exist
    import random
    container_id = f"circuit-equations-{random.randint(10000, 99999)}"

    html = f'''
    <div id="{container_id}">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/temml@0.10.29/dist/Temml-Local.css">
        <style>
            #{container_id} {{
                font-family: 'Computer Modern', 'Latin Modern', Georgia, serif;
                max-width: 100%;
                padding: 1.2em;
                background: #fafbfc;
                border-radius: 10px;
                border: 1px solid #d0d7de;
                line-height: 1.6;
            }}
            #{container_id} .eq-header {{
                text-align: center;
                font-size: 1.3em;
                margin-bottom: 1.5em;
                padding: 0.7em 1em;
                background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
                border-radius: 8px;
                border: 1px solid #c8e6c9;
            }}
            #{container_id} .eq-section {{
                margin-bottom: 1.5em;
                padding: 1em 1.2em;
                background: white;
                border-radius: 8px;
                border-left: 4px solid #1976d2;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            }}
            #{container_id} .eq-section-title {{
                color: #1565c0;
                font-size: 1.05em;
                font-weight: 600;
                margin: 0 0 0.6em 0;
                padding-bottom: 0.4em;
                border-bottom: 1px solid #e3f2fd;
            }}
            #{container_id} .eq-line {{
                margin: 0.5em 0;
                padding: 0.4em 0.8em;
                overflow-x: auto;
                border-radius: 4px;
                transition: background 0.15s;
            }}
            #{container_id} .eq-line:hover {{
                background: #f5f7fa;
            }}
            #{container_id} .eq-math {{
                display: block;
                text-align: left;
            }}
            #{container_id} .eq-error {{
                color: #d32f2f;
                font-family: monospace;
                font-size: 0.85em;
                padding: 0.3em;
                background: #ffebee;
                border-radius: 3px;
            }}
        </style>

        {"".join(equation_blocks)}

        <script>
            (function() {{
                function renderEquations() {{
                    var container = document.getElementById('{container_id}');
                    if (!container) return;

                    var mathSpans = container.querySelectorAll('.eq-math');
                    mathSpans.forEach(function(span) {{
                        var tex = span.getAttribute('data-tex');
                        if (!tex || span.getAttribute('data-rendered')) return;

                        try {{
                            temml.render(tex, span, {{
                                displayMode: true,
                                throwOnError: false,
                                trust: true,
                                strict: false
                            }});
                            span.setAttribute('data-rendered', 'true');
                        }} catch(e) {{
                            span.innerHTML = '<span class="eq-error">' + tex + '</span>';
                            console.warn('Temml render error:', e.message, 'for:', tex);
                        }}
                    }});
                }}

                // Try to render immediately if temml is loaded
                if (typeof temml !== 'undefined') {{
                    renderEquations();
                }} else {{
                    // Load temml and then render
                    var script = document.createElement('script');
                    script.src = 'https://cdn.jsdelivr.net/npm/temml@0.10.29/dist/temml.min.js';
                    script.onload = function() {{
                        renderEquations();
                    }};
                    document.head.appendChild(script);
                }}
            }})();
        </script>
    </div>
    '''
    return html


# =============================================================================
# Rewritten: build_equation_viewer_tab — now with Temml equations FIRST
# =============================================================================

def build_equation_viewer_tab(state: dict):
    """
    Build the Gradio tab for the interactive circuit equation viewer.
    Now renders Temml equations as the PRIMARY output, with plots below.

    Usage in grok.py:
        with gr.TabItem("🔢 Circuit Equations (Live)"):
            build_equation_viewer_tab(state)
    """
    import gradio as gr

    gr.Markdown("### 🔢 Interactive Circuit Equation Viewer")
    gr.Markdown(
        "Enter any two numbers to see the **full computation** as rendered equations. "
        "Every step from embedding through attention, MLP (trig identity), to final logits — "
        "shown as proper math with your numbers substituted in."
    )

    with gr.Row():
        eq_a = gr.Number(value=7, label="Input a", precision=0)
        eq_b = gr.Number(value=13, label="Input b", precision=0)
        eq_run_btn = gr.Button("🔢 Trace Full Computation", variant="primary", size="lg")

    with gr.Row():
        eq_show_abstract = gr.Checkbox(value=True, label="Show abstract equations")
        eq_show_concrete = gr.Checkbox(value=True, label="Show concrete (substituted)")
        eq_max_neurons = gr.Slider(minimum=1, maximum=20, value=5, step=1,
                                    label="Max neurons shown per frequency")

    with gr.Row():
        eq_freq_select = gr.Dropdown(
            choices=[], label="Zoom into frequency (optional)",
            interactive=True, value=None,
        )

    gr.Markdown("---")

    # PRIMARY OUTPUT: Temml-rendered equations
    gr.Markdown("#### Rendered Circuit Equations")
    eq_temml_html = gr.HTML(label="Circuit Equations (Temml)")

    # SECONDARY: Summary
    eq_summary_md = gr.Markdown(label="Computation Summary")

    # TERTIARY: Plots (kept from before)
    gr.Markdown("---")
    gr.Markdown("#### Visual Plots")
    eq_flow_plot = gr.Plot(label="Full Circuit Flow (Visual)")

    # Per-frequency detail
    gr.Markdown("#### Per-Frequency Neuron Detail")
    eq_freq_plot = gr.Plot(label="Frequency Detail")

    # Text trace (collapsible)
    with gr.Accordion("Raw Text Trace (for copy/paste)", open=False):
        eq_text_output = gr.Textbox(
            label="Step-by-Step Equations (Text)",
            lines=30,
            interactive=False
        )

    # Comparison section
    gr.Markdown("---")
    gr.Markdown("#### Compare Multiple Inputs")
    eq_compare_input = gr.Textbox(
        label="Pairs to compare (format: a1,b1; a2,b2; ...)",
        placeholder="7,13; 50,63; 100,13; 56,57",
        value="7,13; 50,63; 100,13; 56,57",
    )
    eq_compare_btn = gr.Button("Compare Inputs", variant="secondary")
    eq_compare_plot = gr.Plot(label="Multi-Input Comparison")

    def run_equation_viewer(a, b, show_abstract, show_concrete, max_neurons, freq_select):
        """Run the full equation viewer with Temml output."""
        if state.get("model") is None:
            empty_html = "<p style='color:red;font-size:1.2em'>⚠️ No model loaded! Train or load a model first.</p>"
            return (
                empty_html,
                "⚠️ No model loaded!",
                go.Figure(),
                go.Figure(),
                "No model available.",
            )

        if state.get("circuit") is None:
            empty_html = "<p style='color:red;font-size:1.2em'>⚠️ No circuit discovered! Run Fourier Discovery first.</p>"
            return (
                empty_html,
                "⚠️ No circuit discovered!",
                go.Figure(),
                go.Figure(),
                "Run Fourier Discovery first.",
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

        # Run the full trace
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

        # Generate Temml HTML (PRIMARY OUTPUT)
        temml_html = generate_temml_equation_html(
            trace,
            show_abstract=show_abstract,
            show_concrete=show_concrete,
            max_neurons_shown=int(max_neurons),
        )

        # Summary markdown
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
            temml_html,
            summary_md,
            result["flow_figure"],
            freq_fig,
            result["equation_text"],
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
        inputs=[eq_a, eq_b, eq_show_abstract, eq_show_concrete, eq_max_neurons, eq_freq_select],
        outputs=[eq_temml_html, eq_summary_md, eq_flow_plot, eq_freq_plot, eq_text_output],
    )

    eq_compare_btn.click(
        fn=run_comparison,
        inputs=[eq_compare_input],
        outputs=[eq_compare_plot],
    )

    # Update frequency dropdown on run
    eq_run_btn.click(
        fn=update_freq_dropdown,
        inputs=[],
        outputs=[eq_freq_select],
    )

# =============================================================================
# Main entry point — combines trace + temml + plots
# =============================================================================

def run_circuit_equation_viewer(model, a: int, b: int,
                                 key_frequencies: list[int],
                                 neuron_assignments: dict,
                                 selected_frequency: Optional[int] = None) -> dict:
    """
    Main entry point for the interactive circuit equation viewer.
    Runs the full trace and generates all outputs.

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
# Visualization: Full Equation Flow as Plotly Figure
# =============================================================================

def make_equation_flow_figure(trace: dict) -> go.Figure:
    """
    Create a multi-panel figure showing the full computation with actual values.
    """
    a = trace["input"]["a"]
    b = trace["input"]["b"]
    P = trace["input"]["P"]
    correct = trace["input"]["correct"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            f"Fourier Components of a={a}, b={b}",
            f"Attention Weights (= → a, b)",
            f"MLP Neurons (pre-ReLU, key only)",
            f"MLP Neurons (post-ReLU, active)",
            f"Logits: Actual vs Ideal Formula",
            f"Probability Distribution",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    steps = trace["steps"]

    # Panel 1: Fourier components
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

    # Panel 2: Attention weights
    attn_data = steps[1]["concrete_values"]["attention_weights_per_head"]
    n_heads = trace["weights_info"]["n_heads"]
    heads = list(range(n_heads))
    fig.add_trace(go.Bar(name="Attn to a", x=[f"Head {h}" for h in heads],
                         y=[attn_data[h]["to_a"] for h in heads],
                         marker_color="blue"), row=1, col=2)
    fig.add_trace(go.Bar(name="Attn to b", x=[f"Head {h}" for h in heads],
                         y=[attn_data[h]["to_b"] for h in heads],
                         marker_color="orange"), row=1, col=2)

    # Panel 3: MLP pre-activations for key neurons
    mlp_step = steps[3]
    neuron_details = mlp_step["key_neuron_details"]
    # Show first 30 key neurons
    shown_neurons = neuron_details[:30]
    if shown_neurons:
        fig.add_trace(go.Bar(
            x=[str(d["neuron_idx"]) for d in shown_neurons],
            y=[d["actual_pre_activation"] for d in shown_neurons],
            marker_color=[f"hsl({(d['frequency']*60)%360}, 70%, 50%)" for d in shown_neurons],
            name="Pre-ReLU",
        ), row=2, col=1)

    # Panel 4: Post-ReLU (active neurons only)
    active_neurons = [d for d in neuron_details if d["actual_post_relu"] > 0][:30]
    if active_neurons:
        fig.add_trace(go.Bar(
            x=[str(d["neuron_idx"]) for d in active_neurons],
            y=[d["actual_post_relu"] for d in active_neurons],
            marker_color=[f"hsl({(d['frequency']*60)%360}, 70%, 50%)" for d in active_neurons],
            name="Post-ReLU (active)",
        ), row=2, col=2)

    # Panel 5: Actual vs Ideal logits
    logit_step = steps[5]
    logits = logit_step["values"]["logits"]
    ideal_logits = logit_step["values"]["ideal_logits_scaled"]

    window = 15
    start = max(0, correct - window)
    end = min(P, correct + window)
    x_range = list(range(start, end))

    fig.add_trace(go.Scatter(
        x=x_range, y=logits[start:end], mode="lines",
        name="Actual logits", line=dict(color="blue", width=2),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=x_range, y=ideal_logits[start:end], mode="lines",
        name="Ideal (formula)", line=dict(color="red", width=2, dash="dash"),
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=[correct], y=[logits[correct]], mode="markers",
        marker=dict(size=12, color="green", symbol="star"),
        name=f"Correct: c={correct}",
    ), row=3, col=1)

    # Panel 6: Probability distribution
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    fig.add_trace(go.Bar(
        x=x_range,
        y=probs[start:end],
        marker_color=["green" if i == correct else "steelblue" for i in x_range],
        name="P(c|a,b)",
    ), row=3, col=2)

    fig.update_layout(
        height=1000,
        width=1100,
        title_text=(
            f"Circuit Trace: ({a} + {b}) mod {P} = {correct} | "
            f"Predicted: {trace['summary']['predicted']} | "
            f"{'✅' if trace['summary']['is_correct'] else '❌'} | "
            f"Conf: {trace['summary']['confidence']*100:.1f}%"
        ),
        showlegend=True,
    )

    return fig


# =============================================================================
# Visualization: Neuron-Level Detail for a Single Frequency
# =============================================================================

def make_neuron_frequency_detail_figure(trace: dict, frequency: int) -> go.Figure:
    """
    Zoom into a single frequency and show every neuron assigned to it.
    """
    a = trace["input"]["a"]
    b = trace["input"]["b"]
    P = trace["input"]["P"]

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
        rows=1, cols=2,
        subplot_titles=(
            f"Pre-ReLU (k={frequency})",
            f"Post-ReLU (k={frequency})",
        ),
    )

    neuron_indices = [d["neuron_idx"] for d in details]
    pre_vals = [d["actual_pre_activation"] for d in details]
    post_vals = [d["actual_post_relu"] for d in details]

    colors_pre = ["green" if v > 0 else "red" for v in pre_vals]
    fig.add_trace(go.Bar(
        x=[str(i) for i in neuron_indices],
        y=pre_vals,
        marker_color=colors_pre,
        name="Pre-ReLU",
    ), row=1, col=1)

    colors_post = ["green" if v > 0 else "lightgray" for v in post_vals]
    fig.add_trace(go.Bar(
        x=[str(i) for i in neuron_indices],
        y=post_vals,
        marker_color=colors_post,
        name="Post-ReLU",
    ), row=1, col=2)

    fig.update_layout(
        height=400,
        title_text=(
            f"Frequency k={frequency} | ω={omega_k:.4f} | "
            f"cos(ω·({a}+{b})) = {ideal_cos:.4f} | "
            f"{neurons_for_freq['firing']}/{neurons_for_freq['total']} fire"
        ),
        showlegend=False,
    )

    return fig


# =============================================================================
# Text-based equation trace
# =============================================================================

def format_equation_trace_text(trace: dict, max_neurons_shown: int = 5) -> str:
    """
    Generate a text-based equation trace for copy/paste.
    """
    a = trace["input"]["a"]
    b = trace["input"]["b"]
    P = trace["input"]["P"]
    correct = trace["input"]["correct"]
    key_freqs = trace["weights_info"]["key_frequencies"]

    lines = []
    lines.append(f"Circuit Trace: ({a} + {b}) mod {P} = {correct}")
    lines.append(f"Key frequencies: {key_freqs}")
    lines.append("")

    # Embedding
    lines.append("=== EMBEDDING ===")
    fourier = trace["steps"][0]["fourier_decomposition"]
    for k in key_freqs:
        f = fourier[k]
        lines.append(
            f"  k={k}: cos(ω·{a})={f['cos_a']:.4f}, sin(ω·{a})={f['sin_a']:.4f}, "
            f"cos(ω·{b})={f['cos_b']:.4f}, sin(ω·{b})={f['sin_b']:.4f}, "
            f"cos(ω·{a+b})={f['cos_apb']:.4f}"
        )
    lines.append("")

    # Attention
    lines.append("=== ATTENTION ===")
    attn_data = trace["steps"][1]["concrete_values"]["attention_weights_per_head"]
    for h in sorted(attn_data.keys()):
        w = attn_data[h]
        lines.append(f"  Head {h}: to_a={w['to_a']:.4f}, to_b={w['to_b']:.4f}")
    lines.append("")

    # MLP
    lines.append("=== MLP ===")
    mlp_step = trace["steps"][3]
    neurons_by_freq = mlp_step["neurons_by_frequency"]
    for k in sorted(neurons_by_freq.keys()):
        freq_info = neurons_by_freq[k]
        omega_k = 2 * np.pi * k / P
        ideal = np.cos(omega_k * (a + b))
        lines.append(f"  k={k}: ideal cos(ω·{a+b})={ideal:.4f}, "
                     f"{freq_info['firing']}/{freq_info['total']} fire")
        for d in freq_info["details"][:max_neurons_shown]:
            status = "FIRE" if d["actual_post_relu"] > 0 else "dead"
            lines.append(
                f"    n{d['neuron_idx']}: trig={d['trig_product']:.4f}, "
                f"pre={d['actual_pre_activation']:.4f} [{status}]"
            )
    lines.append("")

    # Logits
    lines.append("=== LOGITS ===")
    logit_step = trace["steps"][5]
    lines.append(f"  Logit[{correct}] = {logit_step['values']['logit_at_correct']:.4f}")
    lines.append(f"  P(correct) = {logit_step['values']['prob_at_correct']:.4f}")
    lines.append(f"  Correlation = {logit_step['values']['correlation_actual_vs_ideal']:.4f}")
    lines.append(f"  Predicted: {logit_step['values']['predicted']}")
    lines.append(f"  Correct: {logit_step['values']['is_correct']}")

    return "\n".join(lines)


# =============================================================================
# Compare multiple inputs
# =============================================================================

def compare_multiple_inputs(model, pairs: list[tuple[int, int]],
                            key_frequencies: list[int],
                            neuron_assignments: dict) -> go.Figure:
    """
    Run multiple (a, b) pairs and show how the circuit behaves for each.
    """
    P = model.P
    n_pairs = len(pairs)

    fig = make_subplots(
        rows=2, cols=n_pairs,
        subplot_titles=[f"({a}+{b})%{P}={(a+b)%P}" for a, b in pairs] * 2,
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
    )

    for col_idx, (a, b) in enumerate(pairs, 1):
        trace = trace_full_computation(model, a, b, key_frequencies, neuron_assignments)
        correct = (a + b) % P

        # Row 1: cos(ωk(a+b)) for each frequency
        fourier = trace["steps"][0]["fourier_decomposition"]
        freqs = sorted(fourier.keys())
        cos_apb = [fourier[k]["cos_apb"] for k in freqs]

        fig.add_trace(go.Bar(
            x=[f"k={k}" for k in freqs],
            y=cos_apb,
            marker_color=["green" if v > 0 else "red" for v in cos_apb],
            showlegend=False,
        ), row=1, col=col_idx)

        # Row 2: Logits around correct answer
        logits = trace["steps"][5]["values"]["logits"]
        window = 8
        start = max(0, correct - window)
        end = min(P, correct + window)
        x_range = list(range(start, end))

        fig.add_trace(go.Bar(
            x=x_range,
            y=logits[start:end],
            marker_color=["green" if i == correct else "steelblue" for i in x_range],
            showlegend=False,
        ), row=2, col=col_idx)

    fig.update_layout(
        height=600,
        width=250 * n_pairs + 100,
        title_text="Comparison: Same Circuit, Different Inputs",
    )

    return fig
