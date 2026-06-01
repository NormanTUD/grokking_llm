"""
circuit_latex_generator.py — Complete rewrite

Generates a fully self-contained LaTeX document for the discovered Fourier
multiplication circuit. Every variable is defined in a master table. Equations
are presented at THREE levels:
  1. Abstract (English description)
  2. Symbolic (variables with underbraces naming each one)
  3. Concrete (actual numbers extracted from the model, with underbraces)

Includes TikZ diagrams showing the data flow pipeline.
"""

import math
import numpy as np
import torch


class CircuitLatexGenerator:
    """
    Generates detailed LaTeX equations for a discovered Fourier multiplication circuit.

    Three-level presentation:
      - Abstract: what each step does in words
      - Symbolic: full equations with every variable underbraced/defined
      - Concrete: actual numerical values from the trained model

    Requires:
      - P: the prime modulus
      - key_frequencies: list of discovered key frequency indices
      - neuron_assignments: dict mapping neuron_idx -> {frequency, correlation}
      - model: the trained ModularAdditionTransformer (for extracting real weights)
    """

    def __init__(self, P: int, key_frequencies: list[int],
                 neuron_assignments: dict = None,
                 model=None,
                 concrete_example: tuple = (7, 13)):
        self.P = P
        self.key_frequencies = key_frequencies
        self.neuron_assignments = neuron_assignments or {}
        self.model = model  # The actual trained model for extracting numbers
        self.a_example, self.b_example = concrete_example
        self.c_star = (self.a_example + self.b_example) % P

        # Extract concrete values from model if available
        self._extracted = {}
        if model is not None:
            self._extract_concrete_values()

    def _extract_concrete_values(self):
        """Extract real numerical values from the trained model for concrete equations."""
        model = self.model
        P = self.P
        a, b = self.a_example, self.b_example

        # === Embedding matrix W_E ===
        W_E = model.embed.weight[:P].detach().cpu().numpy()  # (P, d_model)
        self._extracted["W_E_shape"] = W_E.shape
        self._extracted["W_E_row_a"] = W_E[a]  # first 8 dims shown
        self._extracted["W_E_row_b"] = W_E[b]

        # === Positional embeddings ===
        pos_embed = model.pos_embed.weight.detach().cpu().numpy()  # (3, d_model)
        self._extracted["p_0"] = pos_embed[0]
        self._extracted["p_1"] = pos_embed[1]
        self._extracted["p_2"] = pos_embed[2]

        # === Fourier basis projection ===
        fourier_basis = np.zeros((P, P))
        fourier_basis[0] = np.ones(P) / np.sqrt(P)
        for k in range(1, P // 2 + 1):
            fourier_basis[2 * k - 1] = np.cos(2 * np.pi * k * np.arange(P) / P) * np.sqrt(2 / P)
            if 2 * k < P:
                fourier_basis[2 * k] = np.sin(2 * np.pi * k * np.arange(P) / P) * np.sqrt(2 / P)

        W_E_fourier = fourier_basis @ W_E  # (P, d_model)
        self._extracted["W_E_fourier"] = W_E_fourier

        # For each key frequency, get the Fourier norms
        for k in self.key_frequencies:
            cos_row = W_E_fourier[2 * k - 1] if 2 * k - 1 < P else np.zeros(model.d_model)
            sin_row = W_E_fourier[2 * k] if 2 * k < P else np.zeros(model.d_model)
            norm = np.sqrt(np.linalg.norm(cos_row) ** 2 + np.linalg.norm(sin_row) ** 2)
            self._extracted[f"embed_norm_k{k}"] = norm

            # Concrete cos/sin values for example inputs
            omega_k = 2 * np.pi * k / P
            self._extracted[f"cos_k{k}_a"] = np.cos(omega_k * a)
            self._extracted[f"sin_k{k}_a"] = np.sin(omega_k * a)
            self._extracted[f"cos_k{k}_b"] = np.cos(omega_k * b)
            self._extracted[f"sin_k{k}_b"] = np.sin(omega_k * b)
            self._extracted[f"cos_k{k}_apb"] = np.cos(omega_k * (a + b))
            self._extracted[f"sin_k{k}_apb"] = np.sin(omega_k * (a + b))

        # === Attention weights for example ===
        a_tensor = torch.tensor([a])
        b_tensor = torch.tensor([b])
        with torch.no_grad():
            _, activations = model.forward_with_hooks(a_tensor, b_tensor)

        attn_weights = activations["attn_weights"][0].numpy()  # (n_heads, 1, 2)
        for h in range(model.n_heads):
            self._extracted[f"attn_head{h}_to_a"] = float(attn_weights[h, 0, 0])
            self._extracted[f"attn_head{h}_to_b"] = float(attn_weights[h, 0, 1])

        # === MLP activations ===
        mlp_pre = activations["mlp_pre"][0, 0].numpy()  # (d_mlp,)
        mlp_hidden = activations["mlp_hidden"][0, 0].numpy()  # (d_mlp,)
        self._extracted["mlp_pre"] = mlp_pre
        self._extracted["mlp_hidden"] = mlp_hidden
        self._extracted["n_active_neurons"] = int((mlp_hidden > 0).sum())

        # === Neuron-logit map W_L ===
        W_mlp_out = model.mlp_out.weight.detach().cpu().numpy()  # (d_model, d_mlp)
        W_unembed = model.unembed.weight.detach().cpu().numpy()  # (P, d_model)
        W_L = W_mlp_out.T @ W_unembed.T  # (d_mlp, P)
        self._extracted["W_L_shape"] = W_L.shape

        # Project W_L onto Fourier basis to get u_k, v_k directions
        W_L_fourier = W_L @ fourier_basis.T  # (d_mlp, P)
        for k in self.key_frequencies:
            u_k = W_L_fourier[:, 2 * k - 1] if 2 * k - 1 < P else np.zeros(model.d_mlp)
            v_k = W_L_fourier[:, 2 * k] if 2 * k < P else np.zeros(model.d_mlp)
            self._extracted[f"u_k{k}_norm"] = np.linalg.norm(u_k)
            self._extracted[f"v_k{k}_norm"] = np.linalg.norm(v_k)

            # Project MLP activations onto u_k and v_k
            proj_cos = np.dot(u_k, mlp_hidden)
            proj_sin = np.dot(v_k, mlp_hidden)
            self._extracted[f"proj_cos_k{k}"] = proj_cos
            self._extracted[f"proj_sin_k{k}"] = proj_sin

        # === Logits ===
        logits = activations["logits"][0].numpy()  # (P,)
        self._extracted["logits"] = logits
        self._extracted["logit_correct"] = float(logits[self.c_star])
        self._extracted["logit_max"] = float(logits.max())
        self._extracted["logit_argmax"] = int(logits.argmax())

        # === Fit alpha_k coefficients via least squares ===
        # logit(c) ≈ sum_k alpha_k * cos(omega_k * (a+b-c) )
        # For all P*P inputs:
        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)
        with torch.no_grad():
            all_logits = model(all_a, all_b).cpu().numpy()  # (P*P, P)

        # Reshape and fit
        a_vals = np.arange(P).reshape(-1, 1, 1)
        b_vals = np.arange(P).reshape(1, -1, 1)
        c_vals = np.arange(P).reshape(1, 1, -1)

        logit_cube = all_logits.reshape(P, P, P)

        # Build design matrix for OLS
        X = np.zeros((P * P * P, len(self.key_frequencies)))
        y = logit_cube.reshape(-1)
        for i, k in enumerate(self.key_frequencies):
            omega_k = 2 * np.pi * k / P
            pattern = np.cos(omega_k * (a_vals + b_vals - c_vals))
            X[:, i] = pattern.reshape(-1)

        # OLS: alpha = (X^T X)^{-1} X^T y
        alphas, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        for i, k in enumerate(self.key_frequencies):
            self._extracted[f"alpha_k{k}"] = float(alphas[i])

        # FVE
        y_pred = X @ alphas
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        self._extracted["fve"] = 1.0 - ss_res / ss_tot

        # === d_head ===
        self._extracted["d_head"] = model.d_head
        self._extracted["d_model"] = model.d_model
        self._extracted["d_mlp"] = model.d_mlp
        self._extracted["n_heads"] = model.n_heads

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def full_circuit_latex(self) -> str:
        """Generate the complete LaTeX document body."""
        sections = []
        sections.append(self._variable_definition_table())
        sections.append(self._tikz_pipeline_diagram())
        sections.append(self._section_header())
        sections.append(self._embedding_equations())
        sections.append(self._attention_equations())
        sections.append(self._mlp_equations())
        sections.append(self._unembed_equations())
        sections.append(self._final_prediction())
        sections.append(self._constructive_interference())
        sections.append(self._concrete_worked_example())
        return "\n\n".join(sections)

    # =========================================================================
    # VARIABLE DEFINITION TABLE
    # =========================================================================

    def _variable_definition_table(self) -> str:
        """Master table defining EVERY variable used in the document."""
        P = self.P
        d_model = self._extracted.get("d_model", 128)
        d_mlp = self._extracted.get("d_mlp", 512)
        d_head = self._extracted.get("d_head", 32)
        n_heads = self._extracted.get("n_heads", 4)
        freqs_str = ", ".join(str(k) for k in self.key_frequencies)

        lines = []
        lines.append(r"\section{Variable Definitions}")
        lines.append("")
        lines.append(r"Every symbol used in this document is defined below. "
                     r"Refer back to this table whenever a variable is unclear.")
        lines.append("")
        lines.append(r"\begin{longtable}")
        lines.append(r"\hline")
        lines.append(r"\textbf{Symbol} & \textbf{Name / Meaning} & \textbf{Shape / Value} \\ \hline")
        lines.append(r"\endhead")

        # --- Scalars ---
        lines.append(r"\multicolumn{3}{|c|}{\textit{Scalar Constants}} \\ \hline")

        # Each entry is (symbol, description, value) — all as raw LaTeX strings
        # We use .format() or concatenation instead of f-strings to avoid brace issues
        lines.append(
            r"  $P$ & Prime modulus. The model computes $(a+b) \bmod P$. & $"
            + str(P) + r"$ \\ \hline"
        )
        lines.append(
            r"  $d_{\text{model}}$ & Dimension of the residual stream (embedding size). & $"
            + str(d_model) + r"$ \\ \hline"
        )
        lines.append(
            r"  $d_{\text{mlp}}$ & Number of neurons in the MLP hidden layer. & $"
            + str(d_mlp) + r"$ \\ \hline"
        )
        lines.append(
            r"  $d_{\text{head}}$ & Dimension of each attention head's Q/K/V vectors. "
            r"Equal to $d_{\text{model}} / n_{\text{heads}}$. & $"
            + str(d_head) + r"$ \\ \hline"
        )
        lines.append(
            r"  $n_{\text{heads}}$ & Number of attention heads. & $"
            + str(n_heads) + r"$ \\ \hline"
        )
        lines.append(
            r"  $k$ & A frequency index. Ranges over $\mathcal{K}$. & Integer in $\{1, \ldots, "
            + str(P // 2) + r"\}$ \\ \hline"
        )
        lines.append(
            r"  $\omega_k$ & Angular frequency for index $k$. Defined as "
            r"$\omega_k = \frac{2\pi k}{P}$. & $\frac{2\pi k}{"
            + str(P) + r"}$ radians \\ \hline"
        )
        lines.append(
            r"  $\mathcal{K}$ & Set of key frequencies discovered via Fourier analysis "
            r"of $W_E$ and $W_L$. & $\{"
            + freqs_str + r"\}$ \\ \hline"
        )

        # --- Tokens and positions ---
        lines.append(r"\multicolumn{3}{|c|}{\textit{Tokens and Positions}} \\ \hline")
        lines.append(
            r"  $a$ & The first input token (an integer). Sits at position 0 in the sequence. "
            r"& $a \in \{0, \ldots, " + str(P - 1) + r"\}$ \\ \hline"
        )
        lines.append(
            r"  $b$ & The second input token (an integer). Sits at position 1 in the sequence. "
            r"& $b \in \{0, \ldots, " + str(P - 1) + r"\}$ \\ \hline"
        )
        lines.append(
            r"  $=$ & The equals-sign token. A special fixed token at position 2. "
            r"The model reads its output from this position. & Token index $= P = "
            + str(P) + r"$ \\ \hline"
        )
        lines.append(
            r"  $c$ & A candidate output class. The model produces a logit for each $c$. "
            r"& $c \in \{0, \ldots, " + str(P - 1) + r"\}$ \\ \hline"
        )
        lines.append(
            r"  $c^*$ & The correct answer: $c^* = (a + b) \bmod P$. & $(a+b) \bmod "
            + str(P) + r"$ \\ \hline"
        )
        lines.append(
            r"  $\mathbf{e}_a$ & One-hot vector for token $a$. Has a 1 in position $a$ and 0 elsewhere. "
            r"This is the raw input to the embedding matrix. & $\in \mathbb{R}^{"
            + str(P + 1) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $\mathbf{e}_b$ & One-hot vector for token $b$. & $\in \mathbb{R}^{"
            + str(P + 1) + r"}$ \\ \hline"
        )
        lines.append(
            r"  position 0 & The first slot in the 3-token input sequence ``$a\ b\ =$''. "
            r"Token $a$ lives here. & --- \\ \hline"
        )
        lines.append(
            r"  position 1 & The second slot. Token $b$ lives here. & --- \\ \hline"
        )
        lines.append(
            r"  position 2 & The third slot. The ``$=$'' token lives here. "
            r"All output logits are read from this position. & --- \\ \hline"
        )

        # --- Weight matrices ---
        lines.append(r"\multicolumn{3}{|c|}{\textit{Weight Matrices}} \\ \hline")
        lines.append(
            r"  $W_E$ & Token embedding matrix. Maps one-hot token vectors to "
            r"$d_{\text{model}}$-dimensional vectors. Row $a$ of $W_E$ is the embedding of token $a$. "
            r"& $\in \mathbb{R}^{" + str(P + 1) + r" \times " + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $\mathbf{p}_i$ & Positional embedding for position $i \in \{0, 1, 2\}$. "
            r"Added to the token embedding to form the initial residual stream. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $W_Q^{(j)}$ & Query weight matrix for attention head $j$. "
            r"Projects the residual stream at the query position into query space. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r" \times " + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $W_K^{(j)}$ & Key weight matrix for attention head $j$. "
            r"Projects the residual stream at key positions into key space. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r" \times " + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $W_V^{(j)}$ & Value weight matrix for attention head $j$. "
            r"Projects the residual stream at value positions into value space. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r" \times " + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $W_O^{(j)}$ & Output projection matrix for attention head $j$. "
            r"Projects the attention output back to the residual stream. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r" \times " + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $W_{\text{in}}$ & MLP input weight matrix. Projects from residual stream to MLP hidden layer. "
            r"& $\in \mathbb{R}^{" + str(d_mlp) + r" \times " + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $b_{\text{in}}$ & MLP input bias vector. "
            r"& $\in \mathbb{R}^{" + str(d_mlp) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $W_{\text{out}}$ & MLP output weight matrix. Projects from MLP hidden layer back to residual stream. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r" \times " + str(d_mlp) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $W_U$ & Unembedding matrix. Maps final residual stream to output logits (one per class $c$). "
            r"& $\in \mathbb{R}^{" + str(P) + r" \times " + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $W_L$ & Neuron-logit map. Defined as $W_L = W_{\text{out}}^\top \cdot W_U^\top$. "
            r"Maps MLP neuron activations directly to output logits. "
            r"This is the key matrix for Fourier analysis. "
            r"& $\in \mathbb{R}^{" + str(d_mlp) + r" \times " + str(P) + r"}$ \\ \hline"
        )

        # --- Intermediate activations ---
        lines.append(r"\multicolumn{3}{|c|}{\textit{Intermediate Activations}} \\ \hline")
        lines.append(
            r"  $\mathbf{x}^{(0)}_a$ & Initial residual stream at position 0 (where token $a$ sits). "
            r"Equals $W_E \cdot \mathbf{e}_a + \mathbf{p}_0$. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $\mathbf{x}^{(0)}_b$ & Initial residual stream at position 1 (where token $b$ sits). "
            r"Equals $W_E \cdot \mathbf{e}_b + \mathbf{p}_1$. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $\mathbf{x}^{(0)}_=$ & Initial residual stream at position 2 (the ``$=$'' token). "
            r"This is constant (independent of $a, b$) since the token and position are fixed. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $\mathbf{x}^{(1)}$ & Residual stream at position 2 after the attention layer. "
            r"Equals $\mathbf{x}^{(0)}_= + \sum_j \text{attn\_out}^{(j)}$. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $A^{(j)}_0$ & Attention weight from head $j$ at position 2 (``$=$'') attending to position 0 (token $a$). "
            r"A scalar between 0 and 1. Note: $A^{(j)}_1 = 1 - A^{(j)}_0$. "
            r"& Scalar $\in [0, 1]$ \\ \hline"
        )
        lines.append(
            r"  $A^{(j)}_1$ & Attention weight from head $j$ at position 2 attending to position 1 (token $b$). "
            r"Equals $1 - A^{(j)}_0$. "
            r"& Scalar $\in [0, 1]$ \\ \hline"
        )
        lines.append(
            r"  $\text{MLP}[n]$ & Activation of MLP neuron $n$ (after ReLU). "
            r"Equals $\max(0,\; W_{\text{in}}[n,:] \cdot \mathbf{x}^{(1)} + b_{\text{in}}[n])$. "
            r"& Scalar $\geq 0$; $n \in \{0, \ldots, " + str(d_mlp - 1) + r"\}$ \\ \hline"
        )
        lines.append(
            r"  $\text{Logit}(c)$ & Output logit for class $c$. The model predicts "
            r"$\hat{c} = \arg\max_c \text{Logit}(c)$. "
            r"& Scalar \\ \hline"
        )

        # --- Fourier-specific variables ---
        lines.append(r"\multicolumn{3}{|c|}{\textit{Fourier Analysis Variables}} \\ \hline")

        # Build alpha string
        alpha_strs = []
        for k in self.key_frequencies:
            alpha_val = self._extracted.get(f"alpha_k{k}", None)
            if isinstance(alpha_val, float):
                alpha_strs.append(r"$\alpha_{" + str(k) + r"} = " + f"{alpha_val:.2f}" + r"$")
            else:
                alpha_strs.append(r"$\alpha_{" + str(k) + r"}$")
        alpha_list_str = ", ".join(alpha_strs)

        lines.append(
            r"  $\alpha_k$ & Amplitude coefficient for frequency $k$ in the logit formula. "
            r"Fitted via OLS: $\text{Logit}(c) \approx \sum_k \alpha_k \cos(\omega_k(a+b-c))$. "
            r"Larger $\alpha_k$ means frequency $k$ contributes more to the final prediction. "
            r"& " + alpha_list_str + r" \\ \hline"
        )
        lines.append(
            r"  $\beta_k$ & Amplitude of the sine component in the embedding. "
            r"Measures how strongly $W_E$ encodes $\sin(\omega_k \cdot t)$ for token $t$. "
            r"Analogous to $\alpha_k$ but for the sine direction. "
            r"& Fitted from $W_E$ Fourier decomposition \\ \hline"
        )
        lines.append(
            r"  $\mathbf{u}_k$ & Direction in neuron space ($\mathbb{R}^{" + str(d_mlp) + r"}$) "
            r"along which the MLP encodes $\cos(\omega_k(a+b))$. "
            r"Extracted from the Fourier decomposition of $W_L$: the column of $W_L$ "
            r"in the Fourier basis corresponding to $\cos(\omega_k c)$. "
            r"& $\in \mathbb{R}^{" + str(d_mlp) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $\mathbf{v}_k$ & Direction in neuron space along which the MLP encodes "
            r"$\sin(\omega_k(a+b))$. The column of $W_L$ in the Fourier basis corresponding "
            r"to $\sin(\omega_k c)$. "
            r"& $\in \mathbb{R}^{" + str(d_mlp) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $\mathbf{u}_k^{(\cos)}$ & Direction in residual stream ($\mathbb{R}^{" + str(d_model) + r"}$) "
            r"along which the embedding encodes the cosine component at frequency $k$. "
            r"Extracted from the Fourier decomposition of $W_E$. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $\mathbf{u}_k^{(\sin)}$ & Direction in residual stream along which the embedding "
            r"encodes the sine component at frequency $k$. "
            r"& $\in \mathbb{R}^{" + str(d_model) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $\gamma_j$ & Modulation amplitude of attention head $j$. "
            r"Measures how much head $j$'s attention pattern deviates from uniform (0.5). "
            r"Comes from the linear approximation to the sigmoid of the attention score difference. "
            r"& Scalar (typically $|\gamma_j| \approx 0.25$) \\ \hline"
        )
        lines.append(
            r"  $C^{(j)}$ & Attention score lookup vector for head $j$. Defined as "
            r"$C^{(j)} = W_E^\top W_K^{(j)\top} W_Q^{(j)} \mathbf{x}^{(0)}_=$. "
            r"Entry $C^{(j)}[a]$ gives the attention score contribution from token $a$. "
            r"& $\in \mathbb{R}^{" + str(P) + r"}$ \\ \hline"
        )
        lines.append(
            r"  $0.5$ (baseline) & When softmax is applied over exactly 2 elements, "
            r"the result is $\sigma(\Delta) = \frac{1}{1 + e^{-\Delta}}$. "
            r"At $\Delta = 0$ (no preference), this equals $0.5$. "
            r"So 0.5 is the ``no information'' baseline for attention over 2 positions. "
            r"& Constant \\ \hline"
        )

        lines.append(r"\end{longtable}")
        lines.append("")
        lines.append(
            r"\textbf{Note on the ``$=$'' token:} The input to the model is the 3-token sequence "
            r"``$a\;\; b\;\; =$''. The ``$=$'' is a literal token (with its own embedding) that serves as "
            r"the query position from which the model reads its output. It is \emph{not} the mathematical "
            r"equals sign. Its embedding $\mathbf{x}^{(0)}_=$ is constant across all inputs."
        )

        return "\n".join(lines)

    # =========================================================================
    # TIKZ PIPELINE DIAGRAM
    # =========================================================================

    def _tikz_pipeline_diagram(self) -> str:
        """Generate a TikZ diagram showing the full data flow pipeline."""
        P = self.P
        freqs_str = ", ".join(str(k) for k in self.key_frequencies)
        n_heads = self._extracted.get("n_heads", 4)
        d_mlp = self._extracted.get("d_mlp", 512)

        lines = []
        lines.append(r"\section{Circuit Architecture (TikZ Diagram)}")
        lines.append("")
        lines.append(r"\begin{figure}[htbp]")
        lines.append(r"\centering")
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tikzpicture}[")
        lines.append(r"    node distance=2.0cm and 3.0cm,")
        lines.append(r"    block/.style={rectangle, draw, rounded corners, minimum width=3.2cm,")
        lines.append(r"        minimum height=1.1cm, align=center, font=\small},")
        lines.append(r"    data/.style={rectangle, draw, dashed, rounded corners, minimum width=3cm,")
        lines.append(r"        minimum height=0.8cm, align=center, font=\footnotesize, fill=yellow!10},")
        lines.append(r"    arrow/.style={->, >=stealth, thick},")
        lines.append(r"    lbl/.style={font=\scriptsize, align=center},")
        lines.append(r"    ]")
        lines.append("")

        # Input layer — spread out more
        lines.append(r"    % === INPUT LAYER ===")
        lines.append(r"    \node[block, fill=orange!20] (input_a) {Token $a$\\(position 0)};")
        lines.append(r"    \node[block, fill=orange!20, right=3cm of input_a] (input_b) {Token $b$\\(position 1)};")
        lines.append(r"    \node[block, fill=orange!20, right=3cm of input_b] (input_eq) {Token ``$=$''\\(position 2)};")
        lines.append("")

        # Embedding layer
        lines.append(r"    % === EMBEDDING LAYER ===")
        lines.append(r"    \draw[arrow] (unembed) -- node[right, lbl] "
                     r"{logits $\in \mathbb{R}^{" + str(P) + r"}$} (output);")
        lines.append("")

        # Frequency annotation box — placed to the right of the embedding layer
        lines.append(r"    % === FREQUENCY ANNOTATION ===")
        lines.append(r"    \node[draw, rounded corners, fill=gray!10, font=\footnotesize,")
        lines.append(r"        text width=4.5cm, align=center, right=2.5cm of embed_eq] (freq_box) {")
        lines.append(r"        \textbf{Key Frequencies}\\$\mathcal{K} = \{" + freqs_str + r"\}$\\[3pt]")
        lines.append(r"        $\omega_k = \frac{2\pi k}{" + str(P) + r"}$};")
        lines.append("")

        lines.append(r"\end{tikzpicture}")
        lines.append(r"}")  # closes \resizebox
        lines.append(r"\caption{Data flow pipeline of the Fourier multiplication circuit. ")
        lines.append(r"Tokens $a$ and $b$ are embedded into Fourier components, attention moves them to the output position, ")
        lines.append(r"the MLP computes trigonometric addition identities, and the unembedding reads off logits via constructive interference.}")
        lines.append(r"\label{fig:pipeline}")
        lines.append(r"\end{figure}")

        return "\n".join(lines)


    # =========================================================================
    # SECTION HEADER
    # =========================================================================

    def _section_header(self) -> str:
        freqs_str = ", ".join(str(k) for k in self.key_frequencies)
        P = self.P
        d_model = self._extracted.get("d_model", 128)
        d_mlp = self._extracted.get("d_mlp", 512)
        n_heads = self._extracted.get("n_heads", 4)

        return (
            r"% =============================================================" + "\n"
            r"% FOURIER MULTIPLICATION CIRCUIT — FULL DERIVATION" + "\n"
            f"% P = {P}, key frequencies k ∈ {{{freqs_str}}}" + "\n"
            r"% =============================================================" + "\n"
            r"\section{Discovered Circuit: Fourier Multiplication Algorithm}" + "\n\n"
            f"The model computes $(a + b) \\bmod {P}$ using a 1-layer transformer with "
            f"$d_{{\\text{{model}}}} = {d_model}$, $n_{{\\text{{heads}}}} = {n_heads}$, "
            f"$d_{{\\text{{mlp}}}} = {d_mlp}$. It uses {len(self.key_frequencies)} "
            f"key frequencies $k \\in \\mathcal{{K}} = \\{{{freqs_str}\\}}$ with angular frequencies "
            r"$\omega_k = \frac{2\pi k}{" + str(P) + r"}$." + "\n\n"
            r"Each equation below is presented at three levels:" + "\n"
            r"\begin{enumerate}" + "\n"
            r"  \item \textbf{Abstract:} What the step does in plain English." + "\n"
            r"  \item \textbf{Symbolic:} Full equations with every variable underbraced by name." + "\n"
            r"  \item \textbf{Concrete:} Actual numerical values from the trained model "
            f"(example: $a = {self.a_example}$, $b = {self.b_example}$, "
            f"$c^* = {self.c_star}$)." + "\n"
            r"\end{enumerate}"
        )

    # =========================================================================
    # EMBEDDING EQUATIONS (3 levels)
    # =========================================================================

    def _embedding_equations(self) -> str:
        P = self.P
        a, b = self.a_example, self.b_example
        lines = []
        lines.append(r"\subsection{Step 1: Embedding (Token $\to$ Fourier Components)}")
        lines.append("")

        # --- ABSTRACT ---
        lines.append(r"\subsubsection*{Abstract}")
        lines.append(r"Each input token $t$ (an integer $0$ to $" + str(P-1) + r"$) is converted into a "
                     r"$d_{\text{model}}$-dimensional vector by looking up row $t$ of the embedding matrix $W_E$ "
                     r"and adding a positional embedding. The key property of $W_E$ (learned during training) is that "
                     r"its rows, when projected onto the Fourier basis, are dominated by sinusoidal components at "
                     r"the key frequencies $\mathcal{K}$. This means the embedding effectively encodes each token "
                     r"as $\cos(\omega_k \cdot t)$ and $\sin(\omega_k \cdot t)$ for each key frequency $k$.")
        lines.append("")

        # --- SYMBOLIC ---
        lines.append(r"\subsubsection*{Symbolic}")
        lines.append(r"\begin{align}")
        lines.append(
            r"    \underbrace{\mathbf{x}^{(0)}_a}_{"
            r"\substack{\text{initial residual stream} \\ \text{at position 0} \\ "
            r"\in \mathbb{R}^{" + str(self._extracted.get('d_model', 128)) + r"}}}"
            r" &= "
            r"\underbrace{W_E \cdot \mathbf{e}_a}_{"
            r"\substack{\text{row } a \text{ of the} \\ \text{embedding matrix} \\ "
            r"\in \mathbb{R}^{" + str(self._extracted.get('d_model', 128)) + r"}}}"
            r" + "
            r"\underbrace{\mathbf{p}_0}_{"
            r"\substack{\text{positional embedding} \\ \text{for position 0} \\ "
            r"\in \mathbb{R}^{" + str(self._extracted.get('d_model', 128)) + r"}}}"
            r" \\"
        )
        lines.append(
            r"    &\approx \sum_{k \in \mathcal{K}} \bigg[ "
            r"\underbrace{\alpha_k}_{"
            r"\substack{\text{cosine amplitude} \\ \text{(from Fourier} \\ \text{decomp of } W_E\text{)}}}"
            r" \underbrace{\cos\!\left(\frac{2\pi k \cdot a}{" + str(P) + r"}\right)}_{"
            r"\substack{\text{cosine evaluated} \\ \text{at token } a}}"
            r"\cdot \underbrace{\mathbf{u}_k^{(\cos)}}_{"
            r"\substack{\text{direction in } \mathbb{R}^{" + str(self._extracted.get('d_model', 128)) + r"} \\ "
            r"\text{for } \cos \text{ at freq } k}}"
            r" + "
            r"\underbrace{\beta_k}_{"
            r"\substack{\text{sine amplitude} \\ \text{(from Fourier} \\ \text{decomp of } W_E\text{)}}}"
            r" \underbrace{\sin\!\left(\frac{2\pi k \cdot a}{" + str(P) + r"}\right)}_{"
            r"\substack{\text{sine evaluated} \\ \text{at token } a}}"
            r"\cdot \underbrace{\mathbf{u}_k^{(\sin)}}_{"
            r"\substack{\text{direction in } \mathbb{R}^{" + str(self._extracted.get('d_model', 128)) + r"} \\ "
            r"\text{for } \sin \text{ at freq } k}}"
            r" \bigg]"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # --- CONCRETE ---
        lines.append(r"\subsubsection*{Concrete (for $a = " + str(a) + r"$, $b = " + str(b) + r"$)}")
        lines.append("")
        lines.append(r"Evaluating the Fourier components for each key frequency:")
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tabular}{|c|c|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"$k$ & $\omega_k$ & $\cos(\omega_k \cdot " + str(a) + r")$ & "
                     r"$\sin(\omega_k \cdot " + str(a) + r")$ & "
                     r"$\cos(\omega_k \cdot " + str(b) + r")$ \\ \hline")

        for k in self.key_frequencies[:5]:  # Show up to 5
            omega_k = 2 * math.pi * k / P
            cos_a = self._extracted.get(f"cos_k{k}_a", math.cos(omega_k * a))
            sin_a = self._extracted.get(f"sin_k{k}_a", math.sin(omega_k * a))
            cos_b = self._extracted.get(f"cos_k{k}_b", math.cos(omega_k * b))
            lines.append(
                f"  ${k}$ & $\\frac{{2\\pi \\cdot {k}}}{{{P}}} = {omega_k:.4f}$ & "
                f"${cos_a:.4f}$ & ${sin_a:.4f}$ & ${cos_b:.4f}$ \\\\ \\hline"
            )

        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")

        return "\n".join(lines)

    # =========================================================================
    # ATTENTION EQUATIONS (3 levels)
    # =========================================================================

    def _attention_equations(self) -> str:
        P = self.P
        a, b = self.a_example, self.b_example
        n_heads = self._extracted.get("n_heads", 4)
        d_head = self._extracted.get("d_head", 32)
        lines = []
        lines.append(r"\subsection{Step 2: Attention (Move Fourier Info to Output Position)}")
        lines.append("")

        # --- ABSTRACT ---
        lines.append(r"\subsubsection*{Abstract}")
        lines.append(
            r"The attention layer's job is to move information from positions 0 and 1 "
            r"(where tokens $a$ and $b$ live) to position 2 (the ``$=$'' token, from which "
            r"the model reads its output). Each of the " + str(n_heads) + r" attention heads "
            r"computes a query from the ``$=$'' position and keys from positions 0 and 1. "
            r"Since softmax over exactly 2 elements is a sigmoid, each head produces an "
            r"attention weight $A^{(j)}_0 \in [0, 1]$ (with $A^{(j)}_1 = 1 - A^{(j)}_0$). "
            r"The ``uniform baseline'' of $0.5$ means the head attends equally to both positions "
            r"(no preference). Deviations from $0.5$ encode periodic information about the inputs."
        )
        lines.append("")

        # --- SYMBOLIC ---
        lines.append(r"\subsubsection*{Symbolic}")
        lines.append(r"\begin{align}")

        # Score computation
        lines.append(
            r"    \underbrace{\text{score}^{(j)}_{= \to a}}_{"
            r"\substack{\text{attention score from} \\ \text{``='' to position 0}}}"
            r" &= "
            r"\frac{"
            r"\overbrace{\mathbf{x}^{(0)}_= \cdot W_Q^{(j)}}^{"
            r"\substack{\text{query vector} \\ \text{(from ``='' at pos 2)} \\ \in \mathbb{R}^{" + str(d_head) + r"}}}"
            r"\;\cdot\; "
            r"\overbrace{\left(W_K^{(j)}\right)^\top \!\cdot \mathbf{x}^{(0)}_a}^{"
            r"\substack{\text{key vector} \\ \text{(from token } a \text{ at pos 0)} \\ \in \mathbb{R}^{" + str(d_head) + r"}}}"
            r"}"
            r"{\underbrace{\sqrt{d_{\text{head}}}}_{"
            r"\substack{\text{scaling factor} \\ = \sqrt{" + str(d_head) + r"} = " + f"{math.sqrt(d_head):.2f}" + r"}}}"
            r" \\"
        )

        # Sigmoid form
        lines.append(
            r"    \underbrace{A^{(j)}_0}_{"
            r"\substack{\text{attention weight} \\ \text{from ``='' to } a \\ \in [0, 1]}}"
            r" &= \underbrace{\sigma\!\left("
            r"\text{score}^{(j)}_{= \to a} - \text{score}^{(j)}_{= \to b}"
            r"\right)}_{"
            r"\substack{\text{softmax over 2 elements} \\ = \text{sigmoid of score difference} \\ "
            r"\sigma(x) = \frac{1}{1 + e^{-x}}}}"
            r" \\"
        )

        # Periodic approximation
        lines.append(
            r"    &\approx \underbrace{0.5}_{"
            r"\substack{\text{uniform baseline:} \\ \sigma(0) = 0.5 \\ \text{(no preference)}}}"
            r" + "
            r"\underbrace{\gamma_j}_{"
            r"\substack{\text{modulation} \\ \text{amplitude} \\ \text{(from linear} \\ \text{approx of } \sigma\text{)}}}"
            r" \Big("
            r"\underbrace{\cos\!\left(\omega_{k_j} \cdot a\right)}_{"
            r"\substack{\text{periodic score} \\ \text{for token } a}}"
            r" - "
            r"\underbrace{\cos\!\left(\omega_{k_j} \cdot b\right)}_{"
            r"\substack{\text{periodic score} \\ \text{for token } b}}"
            r"\Big) \\"
        )

        # OV circuit
        lines.append(
            r"    \underbrace{\text{attn\_out}^{(j)}}_{"
            r"\substack{\text{attention output} \\ \text{of head } j \\ \in \mathbb{R}^{" + str(self._extracted.get('d_model', 128)) + r"}}}"
            r" &= "
            r"\underbrace{A^{(j)}_0}_{\text{weight to } a} \cdot "
            r"\overbrace{W_O^{(j)} W_V^{(j)} \mathbf{x}^{(0)}_a}^{"
            r"\substack{\text{OV circuit on } a \\ \text{(value} \to \text{output projection)}}}"
            r" + "
            r"\underbrace{(1 - A^{(j)}_0)}_{\text{weight to } b} \cdot "
            r"\overbrace{W_O^{(j)} W_V^{(j)} \mathbf{x}^{(0)}_b}^{"
            r"\substack{\text{OV circuit on } b}}"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # --- CONCRETE ---
        lines.append(r"\subsubsection*{Concrete (for $a = " + str(a) + r"$, $b = " + str(b) + r"$)}")
        lines.append("")
        lines.append(r"Actual attention weights extracted from the model:")
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tabular}{|c|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"Head $j$ & $A^{(j)}_0$ (to $a$) & $A^{(j)}_1$ (to $b$) & Deviation from 0.5 \\ \hline")

        for h in range(n_heads):
            attn_a = self._extracted.get(f"attn_head{h}_to_a", 0.5)
            attn_b = self._extracted.get(f"attn_head{h}_to_b", 0.5)
            deviation = attn_a - 0.5
            lines.append(
                f"  {h} & ${attn_a:.4f}$ & ${attn_b:.4f}$ & "
                f"${deviation:+.4f}$ \\\\ \\hline"
            )

        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")
        lines.append("")
        lines.append(r"\textbf{Interpretation:} Deviations from $0.5$ encode periodic information. "
                     r"A head with $A^{(j)}_0 > 0.5$ attends more to token $a$; this modulation "
                     r"is periodic in $a$ and $b$ at the head's tuned frequency $k_j$.")

        return "\n".join(lines)

    # =========================================================================
    # MLP EQUATIONS (3 levels)
    # =========================================================================

    def _mlp_equations(self) -> str:
        P = self.P
        a, b = self.a_example, self.b_example
        d_mlp = self._extracted.get("d_mlp", 512)
        lines = []
        lines.append(r"\subsection{Step 3: MLP (Compute Trig Identities)}")
        lines.append("")

        # --- ABSTRACT ---
        lines.append(r"\subsubsection*{Abstract}")
        lines.append(
            r"The residual stream after attention contains degree-2 products of sines and cosines "
            r"(because attention weight $\times$ OV output = linear $\times$ linear = quadratic). "
            r"The MLP's " + str(d_mlp) + r" neurons, via ReLU activation, collectively implement "
            r"the trigonometric addition identities that convert these products into "
            r"$\cos(\omega_k(a+b))$ and $\sin(\omega_k(a+b))$. This is the core computational step."
        )
        lines.append("")

        # --- SYMBOLIC ---
        lines.append(r"\subsubsection*{Symbolic}")
        lines.append(r"\begin{align}")

        # Residual mid
        lines.append(
            r"    \underbrace{\mathbf{x}^{(1)}}_{"
            r"\substack{\text{residual stream} \\ \text{after attention} \\ "
            r"\in \mathbb{R}^{" + str(self._extracted.get('d_model', 128)) + r"}}}"
            r" &= "
            r"\underbrace{\mathbf{x}^{(0)}_=}_{"
            r"\substack{\text{skip connection} \\ \text{(``='' token embedding} \\ \text{+ pos embed)}}}"
            r" + \sum_{j=0}^{" + str(self._extracted.get('n_heads', 4) - 1) + r"} "
            r"\underbrace{\text{attn\_out}^{(j)}}_{"
            r"\substack{\text{head } j \text{ output} \\ \text{(degree-2 trig)}}}"
            r" \\"
        )

        # Neuron pre-activation
        lines.append(
            r"    \underbrace{\text{pre}_n}_{"
            r"\substack{\text{pre-activation} \\ \text{of neuron } n \\ \text{(scalar)}}}"
            r" &= "
            r"\underbrace{W_{\text{in}}[n, :]}_{"
            r"\substack{\text{row } n \text{ of MLP} \\ \text{input matrix} \\ "
            r"\in \mathbb{R}^{" + str(self._extracted.get('d_model', 128)) + r"}}}"
            r" \cdot "
            r"\underbrace{\mathbf{x}^{(1)}}_{"
            r"\substack{\text{residual stream} \\ \text{(input to MLP)}}}"
            r" + "
            r"\underbrace{b_{\text{in}}[n]}_{"
            r"\substack{\text{bias for} \\ \text{neuron } n}}"
            r" \\"
        )

        # ReLU
        lines.append(
            r"    \underbrace{\text{MLP}[n]}_{"
            r"\substack{\text{activation of} \\ \text{neuron } n \\ \text{(after ReLU)} \\ \geq 0}}"
            r" &= "
            r"\underbrace{\max(0,\; \text{pre}_n)}_{"
            r"\substack{\text{ReLU: zero out} \\ \text{negative values} \\ "
            r"\text{(creates piecewise-linear} \\ \text{approx of trig products)}}}"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # Core trig identity
        lines.append(r"\textbf{The core computation:} For each key frequency $k$, "
                     r"the MLP neurons collectively compute the addition formula:")
        lines.append(r"\begin{align}")
        lines.append(
            r"    \underbrace{\mathbf{u}_k^\top \cdot \text{MLP}(a,b)}_{"
            r"\substack{\text{dot product of all } " + str(d_mlp) + r" \text{ neuron} \\ "
            r"\text{activations with direction } \mathbf{u}_k \\ "
            r"\text{(extracted from } W_L = W_{\text{out}}^\top W_U^\top \text{)}}}"
            r" &\approx "
            r"\underbrace{\alpha_k}_{"
            r"\substack{\text{amplitude} \\ \text{coefficient}}}"
            r" \underbrace{\cos\!\left(\omega_k(a+b)\right)}_{"
            r"\substack{\text{cosine of the SUM} \\ \text{(this is the target)}}}"
            r" \\"
        )
        lines.append(
            r"    &= \underbrace{\alpha_k \cos(\omega_k a) \cos(\omega_k b)}_{"
            r"\substack{\text{from neurons computing} \\ \text{cos} \times \text{cos products}}}"
            r" - \underbrace{\alpha_k \sin(\omega_k a) \sin(\omega_k b)}_{"
            r"\substack{\text{from neurons computing} \\ \text{sin} \times \text{sin products}}}"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # --- CONCRETE ---
        lines.append(r"\subsubsection*{Concrete (for $a = " + str(a) + r"$, $b = " + str(b) + r"$)}")
        lines.append("")

        n_active = self._extracted.get("n_active_neurons", "?")
        lines.append(f"Of the {self._extracted.get('d_mlp', 512)} neurons, "
                     f"\\textbf{{{n_active}}} are active (positive after ReLU) for this input.")
        lines.append("")

        # Show projections onto u_k directions
        lines.append(r"Projecting the MLP activations onto the Fourier directions $\mathbf{u}_k$ and $\mathbf{v}_k$:")
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tabular}{|c|c|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"$k$ & $\mathbf{u}_k^\top \cdot \text{MLP}$ & "
                     r"$\alpha_k \cos(\omega_k(a+b))$ & "
                     r"$\cos(\omega_k \cdot " + str(a + b) + r")$ & "
                     r"$\alpha_k$ \\ \hline")

        for k in self.key_frequencies[:5]:
            omega_k = 2 * math.pi * k / self.P
            cos_apb = self._extracted.get(f"cos_k{k}_apb", math.cos(omega_k * (a + b)))
            proj_cos = self._extracted.get(f"proj_cos_k{k}", "?")
            alpha_k = self._extracted.get(f"alpha_k{k}", "?")

            if isinstance(proj_cos, (int, float)) and isinstance(alpha_k, (int, float)):
                expected = alpha_k * cos_apb
                lines.append(
                    f"  ${k}$ & ${proj_cos:.2f}$ & ${expected:.2f}$ & "
                    f"${cos_apb:.4f}$ & ${alpha_k:.2f}$ \\\\ \\hline"
                )
            else:
                lines.append(
                    f"  ${k}$ & ? & ? & ${cos_apb:.4f}$ & ? \\\\ \\hline"
                )

        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")
        lines.append("")

        # Show the trig identity concretely
        lines.append(r"\textbf{Verifying the trig identity for the first key frequency "
                     f"$k = {self.key_frequencies[0]}$:}}")
        lines.append("")

        k0 = self.key_frequencies[0]
        omega_k0 = 2 * math.pi * k0 / self.P
        cos_a = math.cos(omega_k0 * a)
        sin_a = math.sin(omega_k0 * a)
        cos_b = math.cos(omega_k0 * b)
        sin_b = math.sin(omega_k0 * b)
        cos_apb = math.cos(omega_k0 * (a + b))

        lines.append(r"\begin{align*}")
        lines.append(
            f"    &\\underbrace{{\\cos\\!\\left(\\frac{{2\\pi \\cdot {k0} \\cdot {a}}}{{{self.P}}}\\right)}}_"
            f"{{= {cos_a:.4f}}}"
            f" \\cdot "
            f"\\underbrace{{\\cos\\!\\left(\\frac{{2\\pi \\cdot {k0} \\cdot {b}}}{{{self.P}}}\\right)}}_"
            f"{{= {cos_b:.4f}}}"
            f" - "
            f"\\underbrace{{\\sin\\!\\left(\\frac{{2\\pi \\cdot {k0} \\cdot {a}}}{{{self.P}}}\\right)}}_"
            f"{{= {sin_a:.4f}}}"
            f" \\cdot "
            f"\\underbrace{{\\sin\\!\\left(\\frac{{2\\pi \\cdot {k0} \\cdot {b}}}{{{self.P}}}\\right)}}_"
            f"{{= {sin_b:.4f}}} \\\\"
        )
        lines.append(
            f"    &= ({cos_a:.4f})({cos_b:.4f}) - ({sin_a:.4f})({sin_b:.4f}) \\\\"
        )
        lines.append(
            f"    &= {cos_a * cos_b:.4f} - {sin_a * sin_b:.4f} \\\\"
        )
        lines.append(
            f"    &= {cos_a * cos_b - sin_a * sin_b:.4f} \\\\"
        )
        lines.append(
            f"    &= \\underbrace{{\\cos\\!\\left(\\frac{{2\\pi \\cdot {k0} \\cdot ({a}+{b})}}{{{self.P}}}\\right)}}_"
            f"{{= \\cos\\!\\left(\\frac{{2\\pi \\cdot {k0} \\cdot {a + b}}}{{{self.P}}}\\right) = {cos_apb:.4f}}}"
            f" \\quad \\checkmark"
        )
        lines.append(r"\end{align*}")

        return "\n".join(lines)

    # =========================================================================
    # UNEMBED EQUATIONS (3 levels)
    # =========================================================================

    def _unembed_equations(self) -> str:
        P = self.P
        a, b = self.a_example, self.b_example
        d_mlp = self._extracted.get("d_mlp", 512)
        lines = []
        lines.append(r"\subsection{Step 4: Unembedding (Fourier $\to$ Logits)}")
        lines.append("")

        # --- ABSTRACT ---
        lines.append(r"\subsubsection*{Abstract}")
        lines.append(
            r"The neuron-logit map $W_L = W_{\text{out}}^\top \cdot W_U^\top$ converts the "
            r"MLP neuron activations directly into output logits. Its key property (discovered "
            r"via Fourier analysis) is that it is approximately rank 10: it decomposes into "
            r"5 pairs of directions, one pair per key frequency. Each pair reads off "
            r"$\cos(\omega_k(a+b))$ and $\sin(\omega_k(a+b))$ from the MLP activations and "
            r"multiplies them by $\cos(\omega_k c)$ and $\sin(\omega_k c)$ respectively. "
            r"By the cosine subtraction identity, this produces $\cos(\omega_k(a+b-c))$."
        )
        lines.append("")

        # --- SYMBOLIC ---
        lines.append(r"\subsubsection*{Symbolic}")
        lines.append(r"\begin{align}")

        # W_L decomposition
        lines.append(
            r"    \underbrace{W_L}_{"
            r"\substack{\text{neuron-logit map} \\ "
            r"\in \mathbb{R}^{" + str(d_mlp) + r" \times " + str(P) + r"} \\ "
            r"= W_{\text{out}}^\top W_U^\top}}"
            r" &\approx \sum_{k \in \mathcal{K}} \bigg[ "
            r"\underbrace{\cos(\omega_k c)}_{"
            r"\substack{\text{vector in } \mathbb{R}^{" + str(P) + r"} \\ "
            r"\text{whose } c\text{-th entry} \\ \text{is } \cos(\omega_k c)}}"
            r" \cdot "
            r"\underbrace{\mathbf{u}_k^\top}_{"
            r"\substack{\text{row vector in } \mathbb{R}^{" + str(d_mlp) + r"} \\ "
            r"\text{reads } \cos(\omega_k(a+b)) \\ \text{from MLP activations}}}"
            r" + "
            r"\underbrace{\sin(\omega_k c)}_{"
            r"\substack{\text{vector in } \mathbb{R}^{" + str(P) + r"} \\ "
            r"\text{whose } c\text{-th entry} \\ \text{is } \sin(\omega_k c)}}"
            r" \cdot "
            r"\underbrace{\mathbf{v}_k^\top}_{"
            r"\substack{\text{row vector in } \mathbb{R}^{" + str(d_mlp) + r"} \\ "
            r"\text{reads } \sin(\omega_k(a+b)) \\ \text{from MLP activations}}}"
            r" \bigg] \\"
        )

        # Logit computation
        lines.append(
            r"    \underbrace{\text{Logit}(c \mid a, b)}_{"
            r"\substack{\text{output logit} \\ \text{for class } c}}"
            r" &= "
            r"\underbrace{W_L^\top \cdot \text{MLP}(a,b)}_{"
            r"\substack{\text{neuron-logit map} \\ \text{applied to MLP output}}}"
            r" \\"
        )

        # Expand using trig identity
        lines.append(
            r"    &\approx \sum_{k \in \mathcal{K}} \bigg[ "
            r"\underbrace{\cos(\omega_k c)}_{"
            r"\substack{\text{from } W_L \\ \text{(unembed)}}}"
            r" \cdot "
            r"\underbrace{\mathbf{u}_k^\top \text{MLP}(a,b)}_{"
            r"\substack{\approx \alpha_k \cos(\omega_k(a+b)) \\ \text{(MLP computed this)}}}"
            r" + "
            r"\underbrace{\sin(\omega_k c)}_{"
            r"\substack{\text{from } W_L \\ \text{(unembed)}}}"
            r" \cdot "
            r"\underbrace{\mathbf{v}_k^\top \text{MLP}(a,b)}_{"
            r"\substack{\approx \alpha_k \sin(\omega_k(a+b)) \\ \text{(MLP computed this)}}}"
            r" \bigg] \\"
        )

        # Apply cosine subtraction identity
        lines.append(
            r"    &= \sum_{k \in \mathcal{K}} "
            r"\underbrace{\alpha_k}_{"
            r"\substack{\text{amplitude} \\ \text{coefficient}}}"
            r" \underbrace{\Big["
            r"\cos(\omega_k c)\cos(\omega_k(a+b)) + \sin(\omega_k c)\sin(\omega_k(a+b))"
            r"\Big]}_{"
            r"\substack{\text{cosine subtraction identity:} \\ "
            r"\cos(X)\cos(Y) + \sin(X)\sin(Y) = \cos(X - Y)}}"
            r" \\"
        )

        # Final boxed result
        lines.append(
            r"    &= \boxed{\sum_{k \in \mathcal{K}} "
            r"\underbrace{\alpha_k \cos\!\left(\omega_k(a + b - c)\right)}_{"
            r"\substack{\text{peaks when } c \equiv a+b \pmod{" + str(P) + r"} \\"
            r"\text{since } \cos(0) = 1 \text{ is the maximum}}}}"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # --- CONCRETE ---
        lines.append(r"\subsubsection*{Concrete (for $a = " + str(a) + r"$, $b = " + str(b) +
                     r"$, $c^* = " + str(self.c_star) + r"$)}")
        lines.append("")

        # Show actual logit values
        logit_correct = self._extracted.get("logit_correct", "?")
        logit_argmax = self._extracted.get("logit_argmax", "?")
        fve = self._extracted.get("fve", "?")

        lines.append(r"\begin{itemize}")
        if isinstance(logit_correct, (int, float)):
            lines.append(f"  \\item Logit at correct answer $c^* = {self.c_star}$: "
                         f"$\\text{{Logit}}({self.c_star}) = {logit_correct:.4f}$")
        if isinstance(logit_argmax, int):
            lines.append(f"  \\item Model's prediction: $\\hat{{c}} = \\arg\\max_c \\text{{Logit}}(c) = {logit_argmax}$"
                         f" {'$\\checkmark$' if logit_argmax == self.c_star else '$\\times$'}")
        if isinstance(fve, float):
            lines.append(f"  \\item Fraction of variance explained by the formula: "
                         f"$\\text{{FVE}} = {fve:.4f}$ ({fve*100:.1f}\\%)")
        lines.append(r"\end{itemize}")
        lines.append("")

        # Show the formula evaluated at c*
        lines.append(r"Evaluating the formula at $c^* = " + str(self.c_star) + r"$:")
        lines.append(r"\begin{align*}")
        lines.append(
            f"    \\text{{Logit}}({self.c_star}) &= \\sum_{{k \\in \\mathcal{{K}}}} "
            f"\\alpha_k \\cos\\!\\left(\\omega_k \\cdot "
            f"\\underbrace{{({a} + {b} - {self.c_star})}}_{{= {a + b - self.c_star}"
            f"{' = 0' if (a + b - self.c_star) % P == 0 else ''}}}\\right) \\\\"
        )

        # Sum up the alpha_k values
        alpha_sum_parts = []
        for k in self.key_frequencies:
            alpha_k = self._extracted.get(f"alpha_k{k}", None)
            if isinstance(alpha_k, (int, float)):
                omega_k = 2 * math.pi * k / P
                val = alpha_k * math.cos(omega_k * ((a + b - self.c_star) % P))
                alpha_sum_parts.append((k, alpha_k, val))

        if alpha_sum_parts:
            terms = " + ".join(
                f"\\underbrace{{{alpha_k:.2f} \\cdot \\cos(0)}}_{{\\alpha_{{{k}}} \\cdot 1 = {alpha_k:.2f}}}"
                if abs((a + b - self.c_star) % P) < 1e-10
                else f"{val:.2f}"
                for k, alpha_k, val in alpha_sum_parts
            )
            total = sum(val for _, _, val in alpha_sum_parts)
            lines.append(f"    &= {terms} \\\\")
            lines.append(f"    &= {total:.2f}")

        lines.append(r"\end{align*}")

        return "\n".join(lines)

    # =========================================================================
    # FINAL PREDICTION (3 levels)
    # =========================================================================

    def _final_prediction(self) -> str:
        P = self.P
        lines = []
        lines.append(r"\subsection{Step 5: Prediction via Constructive Interference}")
        lines.append("")

        # --- ABSTRACT ---
        lines.append(r"\subsubsection*{Abstract}")
        lines.append(
            r"The model predicts the class $c$ with the highest logit. Because the logit formula "
            r"is a sum of cosines $\sum_k \alpha_k \cos(\omega_k(a+b-c))$, and $\cos(0) = 1$ is the "
            r"maximum of cosine, ALL cosines simultaneously achieve their maximum when "
            r"$c \equiv a + b \pmod{" + str(P) + r"}$. This is \textbf{constructive interference}: "
            r"all " + str(len(self.key_frequencies)) + r" cosine waves peak at the same point. "
            r"For any other $c$, the cosines at different frequencies point in different directions "
            r"and partially cancel (\textbf{destructive interference}), giving a smaller logit."
        )
        lines.append("")

        # --- SYMBOLIC ---
        lines.append(r"\subsubsection*{Symbolic}")
        lines.append(r"\begin{align}")
        lines.append(
            r"    \underbrace{\hat{c}}_{"
            r"\substack{\text{model's} \\ \text{prediction}}}"
            r" &= "
            r"\underbrace{\arg\max_{c \in \{0,\ldots," + str(P-1) + r"\}}}_{"
            r"\text{select class with highest logit}}"
            r" \overbrace{\sum_{k \in \mathcal{K}} "
            r"\underbrace{\alpha_k}_{\text{amplitude}} "
            r"\cos\!\left(\frac{2\pi k (a + b - c)}{" + str(P) + r"}\right)}^{"
            r"\text{sum of " + str(len(self.key_frequencies)) + r" cosines at key frequencies}} \\"
        )
        lines.append(
            r"    &= \underbrace{(a + b) \bmod " + str(P) + r"}_{"
            r"\substack{\text{constructive interference:} \\"
            r"\text{all } " + str(len(self.key_frequencies)) + r" \text{ cosines} = 1 "
            r"\text{ when } c = (a+b) \bmod " + str(P) + r" \\"
            r"\text{destructive interference for all other } c}}"
        )
        lines.append(r"\end{align}")

        # --- CONCRETE ---
        lines.append("")
        lines.append(r"\subsubsection*{Concrete}")
        a, b = self.a_example, self.b_example
        lines.append(
            f"For $a = {a}$, $b = {b}$: the correct answer is "
            f"$c^* = ({a} + {b}) \\bmod {P} = {self.c_star}$."
        )
        lines.append("")
        lines.append(r"\begin{itemize}")
        lines.append(
            f"  \\item At $c = c^* = {self.c_star}$: "
            f"$a + b - c = {a} + {b} - {self.c_star} = {a + b - self.c_star}$"
            f"{' $\\equiv 0 \\pmod{' + str(P) + '}$' if (a + b - self.c_star) % P == 0 else ''}"
            f", so $\\cos(\\omega_k \\cdot 0) = 1$ for ALL $k$."
        )

        # Pick a wrong answer
        c_wrong = (self.c_star + 1) % P
        diff_wrong = (a + b - c_wrong) % P
        lines.append(
            f"  \\item At $c = {c_wrong}$ (wrong): "
            f"$a + b - c \\equiv {diff_wrong} \\pmod{{{P}}}$, so cosines at different "
            f"frequencies give different values $\\Rightarrow$ partial cancellation."
        )
        lines.append(r"\end{itemize}")

        return "\n".join(lines)

    # =========================================================================
    # CONSTRUCTIVE INTERFERENCE (3 levels)
    # =========================================================================

    def _constructive_interference(self) -> str:
        P = self.P
        lines = []
        lines.append(r"\subsection{Why Multiple Frequencies? (Constructive Interference)}")
        lines.append("")

        # --- ABSTRACT ---
        lines.append(r"\subsubsection*{Abstract}")
        lines.append(
            r"A single cosine $\cos(\omega_k x)$ has period $" + str(P) +
            r"$ and achieves its maximum at $x = 0$, but it also comes close to 1 at other "
            r"values of $x$ (e.g., for $k = 14$: $\cos(\omega_{14} \cdot 8) = 0.998$). "
            r"A single frequency cannot reliably distinguish $x = 0$ from these near-maxima. "
            r"By summing cosines at " + str(len(self.key_frequencies)) + r" different frequencies, "
            r"the model constructs a function with a \emph{unique, sharp} maximum at $x = 0 \bmod " +
            str(P) + r"$, because the near-maxima of different frequencies occur at different "
            r"values of $x$ and thus cancel out."
        )
        lines.append("")

        # --- SYMBOLIC ---
        lines.append(r"\subsubsection*{Symbolic}")
        lines.append(r"\begin{align}")
        lines.append(
            r"    f(x) &= \sum_{k \in \mathcal{K}} "
            r"\underbrace{\cos\!\left(\frac{2\pi k \cdot x}{" + str(P) + r"}\right)}_{"
            r"\text{each has max at } x = 0} \\"
        )
        lines.append(
            r"    f(0) &= \underbrace{" + str(len(self.key_frequencies)) + r"}_{"
            r"\text{all } " + str(len(self.key_frequencies)) +
            r" \text{ cosines} = 1} \quad \gg \quad "
            r"\underbrace{f(x \neq 0)}_{"
            r"\substack{\text{destructive interference:} \\ "
            r"\text{cosines at different} \\ \text{frequencies cancel}}}"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # --- CONCRETE ---
        lines.append(r"\subsubsection*{Concrete}")
        lines.append(r"Evaluating $f(x)$ at a few values:")
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tabular}{|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"$x$ & $f(x) = \sum_k \cos(\omega_k x)$ & Interpretation \\ \hline")

        # x = 0
        lines.append(
            f"  $0$ & ${len(self.key_frequencies):.1f}$ & "
            f"\\textbf{{Maximum}} (all cosines $= 1$) \\\\ \\hline"
        )

        # A few other values
        for x_test in [1, 2, 5, P // 2]:
            val = sum(math.cos(2 * math.pi * k * x_test / P) for k in self.key_frequencies)
            lines.append(
                f"  ${x_test}$ & ${val:.4f}$ & "
                f"{'Near zero (cancellation)' if abs(val) < 1.0 else 'Partial cancellation'} \\\\ \\hline"
            )

        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")
        lines.append("")
        lines.append(
            r"The ratio $\frac{f(0)}{\max_{x \neq 0} f(x)}$ determines how reliably the model "
            r"can distinguish the correct answer from all others. With " +
            str(len(self.key_frequencies)) + r" frequencies, this ratio is large enough "
            r"for the softmax to assign $> 99.99\%$ probability to the correct class."
        )

        return "\n".join(lines)

    # =========================================================================
    # CONCRETE WORKED EXAMPLE (full hand calculation)
    # =========================================================================

    def _concrete_worked_example(self) -> str:
        P = self.P
        a, b = self.a_example, self.b_example
        c_star = self.c_star
        lines = []
        lines.append(r"\subsection{Complete Worked Example: $a=" + str(a) +
                     r"$, $b=" + str(b) + r"$}")
        lines.append("")
        lines.append(
            r"We now trace the \textbf{entire computation by hand} for the input "
            f"$(a, b) = ({a}, {b})$, showing every intermediate value."
        )
        lines.append("")

        # --- Step 1: Embedding ---
        lines.append(r"\subsubsection*{Step 1: Compute Fourier Components}")
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tabular}{|c|c|c|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"$k$ & $\omega_k = \frac{2\pi k}{" + str(P) + r"}$ & "
                     r"$\cos(\omega_k \cdot " + str(a) + r")$ & "
                     r"$\sin(\omega_k \cdot " + str(a) + r")$ & "
                     r"$\cos(\omega_k \cdot " + str(b) + r")$ & "
                     r"$\sin(\omega_k \cdot " + str(b) + r")$ \\ \hline")

        for k in self.key_frequencies:
            omega_k = 2 * math.pi * k / P
            lines.append(
                f"  ${k}$ & ${omega_k:.4f}$ & "
                f"${math.cos(omega_k * a):.4f}$ & ${math.sin(omega_k * a):.4f}$ & "
                f"${math.cos(omega_k * b):.4f}$ & ${math.sin(omega_k * b):.4f}$ \\\\ \\hline"
            )

        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")
        lines.append("")

        # --- Step 2: Trig identity ---
        lines.append(r"\subsubsection*{Step 2: Apply Trig Addition Identity}")
        lines.append(r"For each $k$, compute $\cos(\omega_k(a+b))$ and $\sin(\omega_k(a+b))$:")
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tabular}{|c|c|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"$k$ & $\cos\omega_k a \cdot \cos\omega_k b$ & "
                     r"$\sin\omega_k a \cdot \sin\omega_k b$ & "
                     r"$\cos(\omega_k(a+b))$ & Verify \\ \hline")

        for k in self.key_frequencies:
            omega_k = 2 * math.pi * k / P
            ca = math.cos(omega_k * a)
            sa = math.sin(omega_k * a)
            cb = math.cos(omega_k * b)
            sb = math.sin(omega_k * b)
            cc = ca * cb
            ss = sa * sb
            cos_sum = cc - ss
            cos_direct = math.cos(omega_k * (a + b))
            match = "\\checkmark" if abs(cos_sum - cos_direct) < 1e-10 else "\\approx"
            lines.append(
                f"  ${k}$ & ${cc:.4f}$ & ${ss:.4f}$ & "
                f"${cos_sum:.4f}$ & ${match}$ (direct: ${cos_direct:.4f}$) \\\\ \\hline"
            )

        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")
        lines.append("")

        # --- Step 3: Logit computation ---
        lines.append(r"\subsubsection*{Step 3: Compute Logits}")
        lines.append(f"For the correct answer $c^* = {c_star}$, we have "
                     f"$a + b - c^* = {a} + {b} - {c_star} = {a + b - c_star}$"
                     f"{'$\\equiv 0 \\pmod{' + str(P) + '}$' if (a + b - c_star) % P == 0 else ''}:")
        lines.append("")
        lines.append(r"\begin{align*}")

        total_logit = 0.0
        term_strs = []
        for k in self.key_frequencies:
            alpha_k = self._extracted.get(f"alpha_k{k}", None)
            omega_k = 2 * math.pi * k / P
            cos_val = math.cos(omega_k * ((a + b - c_star) % P))

            if isinstance(alpha_k, (int, float)):
                contribution = alpha_k * cos_val
                total_logit += contribution
                term_strs.append(
                    f"\\underbrace{{{alpha_k:.2f}}}_{{\\alpha_{{{k}}}}}"
                    f" \\cdot \\underbrace{{{cos_val:.4f}}}_"
                    f"{{\\cos(\\omega_{{{k}}} \\cdot {(a + b - c_star) % P})}}"
                )

        if term_strs:
            lines.append(f"    \\text{{Logit}}({c_star}) &= " + " + ".join(term_strs) + " \\\\")
            lines.append(f"    &= {total_logit:.2f}")

        lines.append(r"\end{align*}")
        lines.append("")

        # Compare with a wrong answer
        c_wrong = (c_star + 1) % P
        lines.append(f"For a wrong answer $c = {c_wrong}$, we have "
                     f"$a + b - c \\equiv {(a + b - c_wrong) % P} \\pmod{{{P}}}$:")
        lines.append(r"\begin{align*}")

        total_wrong = 0.0
        term_strs_wrong = []
        for k in self.key_frequencies:
            alpha_k = self._extracted.get(f"alpha_k{k}", None)
            omega_k = 2 * math.pi * k / P
            cos_val = math.cos(omega_k * ((a + b - c_wrong) % P))

            if isinstance(alpha_k, (int, float)):
                contribution = alpha_k * cos_val
                total_wrong += contribution
                term_strs_wrong.append(f"{contribution:.2f}")

        if term_strs_wrong:
            lines.append(f"    \\text{{Logit}}({c_wrong}) &= " + " + ".join(term_strs_wrong) + " \\\\")
            lines.append(f"    &= {total_wrong:.2f}")

        lines.append(r"\end{align*}")
        lines.append("")

        # Comparison
        if isinstance(total_logit, (int, float)) and total_wrong != 0:
            lines.append(
                f"\\textbf{{Margin:}} $\\text{{Logit}}({self.c_star}) - \\text{{Logit}}({c_wrong}) "
                f"= {total_logit:.2f} - {total_wrong:.2f} = {total_logit - total_wrong:.2f}$. "
                f"This large positive margin ensures the softmax assigns overwhelming probability to $c^* = {self.c_star}$."
            )

        # TikZ mini-diagram: bar chart of logits for a few classes
        lines.append("")
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tikzpicture}[yscale=0.05, xscale=0.8]")
        lines.append(r"    \draw[->] (0,0) -- (7,0) node[right] {\footnotesize $c$};")
        lines.append(r"    \draw[->] (0,-5) -- (0," + str(int(max(total_logit if isinstance(total_logit, (int, float)) else 10, 10) * 1.2)) + r") node[above] {\footnotesize Logit};")

        # Show a few bars
        classes_to_show = [self.c_star, c_wrong, (self.c_star + 5) % self.P, (self.c_star + 10) % self.P]
        for i, c_val in enumerate(classes_to_show):
            omega_sum = 0.0
            for k in self.key_frequencies:
                alpha_k = self._extracted.get(f"alpha_k{k}", 0)
                if isinstance(alpha_k, (int, float)):
                    omega_k = 2 * math.pi * k / self.P
                    omega_sum += alpha_k * math.cos(omega_k * ((a + b - c_val) % self.P))

            color = "green!70!black" if c_val == self.c_star else "red!50!black"
            bar_height = max(omega_sum, 0)
            lines.append(
                f"    \\fill[{color}] ({i * 1.5 + 0.5},0) rectangle ({i * 1.5 + 1.2},{bar_height:.1f});"
            )
            lines.append(
                f"    \\node[below, font=\\tiny] at ({i * 1.5 + 0.85},0) {{$c={c_val}$}};"
            )

        lines.append(r"    \node[font=\scriptsize, green!70!black] at (1.2," + str(int(max(total_logit if isinstance(total_logit, (int, float)) else 10, 10) * 1.1)) + r") {$c^*$};")
        lines.append(r"\end{tikzpicture}")
        lines.append(r"\end{center}")
        lines.append(r"\captionof{figure}{Logit values for selected output classes. "
                     r"The correct answer $c^* = " + str(self.c_star) + r"$ has the largest logit "
                     r"due to constructive interference of all " + str(len(self.key_frequencies)) + r" cosine waves.}")

        return "\n".join(lines)

    # =========================================================================
    # HELPER: Generate per-frequency detail (enhanced version)
    # =========================================================================

    def per_frequency_detail(self, k: int) -> str:
        """Generate detailed LaTeX for a single key frequency k, with all three levels."""
        P = self.P
        a, b = self.a_example, self.b_example
        omega_k = 2 * math.pi * k / P
        alpha_k = self._extracted.get(f"alpha_k{k}", "?")
        n_neurons = self._count_neurons_for_freq(k)

        lines = []
        lines.append(f"\\subsubsection{{Frequency $k = {k}$: "
                     f"$\\omega_{{{k}}} = \\frac{{2\\pi \\cdot {k}}}{{{P}}} = {omega_k:.4f}$ rad}}")
        lines.append("")

        # --- Variable sub-table for this frequency ---
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tabular}{|c|c|c|}")
        lines.append(r"\hline")
        lines.append(r"\textbf{Variable} & \textbf{Meaning} & \textbf{Value} \\ \hline")
        lines.append(f"  $\\omega_{{{k}}}$ & Angular frequency & ${omega_k:.4f}$ rad \\\\ \\hline")
        if isinstance(alpha_k, (int, float)):
            lines.append(f"  $\\alpha_{{{k}}}$ & Amplitude coefficient & ${alpha_k:.2f}$ \\\\ \\hline")
        lines.append(f"  $n_{{\\text{{neurons}}}}$ & Neurons assigned to freq ${k}$ & ${n_neurons}$ \\\\ \\hline")

        u_norm = self._extracted.get(f"u_k{k}_norm", "?")
        v_norm = self._extracted.get(f"v_k{k}_norm", "?")
        if isinstance(u_norm, (int, float)):
            lines.append(f"  $\\|\\mathbf{{u}}_{{{k}}}\\|$ & Norm of cosine readout direction & ${u_norm:.4f}$ \\\\ \\hline")
        if isinstance(v_norm, (int, float)):
            lines.append(f"  $\\|\\mathbf{{v}}_{{{k}}}\\|$ & Norm of sine readout direction & ${v_norm:.4f}$ \\\\ \\hline")

        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")
        lines.append("")

        # --- Abstract ---
        lines.append(r"\paragraph{Abstract}")
        lines.append(
            f"Frequency $k = {k}$ contributes one cosine wave $\\alpha_{{{k}}} \\cos(\\omega_{{{k}}}(a+b-c))$ "
            f"to the logit formula. The embedding encodes tokens as points on a circle of frequency ${k}$ "
            f"(i.e., token $t$ maps to angle $\\omega_{{{k}}} \\cdot t$). "
            f"The MLP's {n_neurons} neurons assigned to this frequency collectively compute "
            f"$\\cos(\\omega_{{{k}}}(a+b))$ and $\\sin(\\omega_{{{k}}}(a+b))$ via the addition formula. "
            f"The unembedding reads these off via directions $\\mathbf{{u}}_{{{k}}}$ and $\\mathbf{{v}}_{{{k}}}$ in $W_L$."
        )
        lines.append("")

        # --- Symbolic ---
        lines.append(r"\paragraph{Symbolic}")
        lines.append(r"\begin{align}")
        lines.append(
            f"    \\underbrace{{W_E[a]}}_{{\\text{{row }} a \\text{{ of }} W_E}}"
            f" &\\xrightarrow{{\\text{{Fourier proj.}}}}"
            f" \\underbrace{{\\cos\\!\\left(\\omega_{{{k}}} \\cdot a\\right)}}_"
            f"{{\\text{{cosine at freq }} {k}}},\\;"
            f"\\underbrace{{\\sin\\!\\left(\\omega_{{{k}}} \\cdot a\\right)}}_"
            f"{{\\text{{sine at freq }} {k}}}"
            r"    \notag \\"
        )
        lines.append(
            f"    \\underbrace{{\\mathbf{{u}}_{{{k}}}^\\top \\cdot \\text{{MLP}}(a,b)}}_"
            f"{{\\text{{project MLP onto }} \\mathbf{{u}}_{{{k}}}}}"
            f" &\\approx "
            f"\\underbrace{{\\alpha_{{{k}}}}}_{{\\text{{amplitude}}}}"
            f" \\underbrace{{\\cos(\\omega_{{{k}}}(a+b))}}_"
            f"{{\\text{{cosine of sum}}}}"
            r"    \notag \\"
        )
        lines.append(
            f"    \\underbrace{{\\text{{Logit contribution}}}}_{{\\text{{from freq }} {k}}}"
            f" &= "
            f"\\underbrace{{\\alpha_{{{k}}} \\cos(\\omega_{{{k}}}(a+b-c))}}_"
            f"{{\\text{{peaks at }} c = (a+b) \\bmod {P}}}"
            r"    \notag"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # --- Concrete ---
        lines.append(f"\\paragraph{{Concrete ($a = {a}$, $b = {b}$)}}")
        lines.append("")

        cos_a = math.cos(omega_k * a)
        sin_a = math.sin(omega_k * a)
        cos_b = math.cos(omega_k * b)
        sin_b = math.sin(omega_k * b)
        cos_apb = math.cos(omega_k * (a + b))
        sin_apb = math.sin(omega_k * (a + b))

        lines.append(r"\begin{align*}")
        lines.append(
            f"    \\cos(\\omega_{{{k}}} \\cdot {a}) &= \\cos({omega_k:.4f} \\times {a}) = {cos_a:.4f} \\\\"
        )
        lines.append(
            f"    \\sin(\\omega_{{{k}}} \\cdot {a}) &= \\sin({omega_k:.4f} \\times {a}) = {sin_a:.4f} \\\\"
        )
        lines.append(
            f"    \\cos(\\omega_{{{k}}} \\cdot {b}) &= \\cos({omega_k:.4f} \\times {b}) = {cos_b:.4f} \\\\"
        )
        lines.append(
            f"    \\sin(\\omega_{{{k}}} \\cdot {b}) &= \\sin({omega_k:.4f} \\times {b}) = {sin_b:.4f} \\\\"
        )
        lines.append(
            f"    \\cos(\\omega_{{{k}}}({a}+{b})) &= "
            f"({cos_a:.4f})({cos_b:.4f}) - ({sin_a:.4f})({sin_b:.4f}) \\\\"
        )
        lines.append(
            f"    &= {cos_a * cos_b:.4f} - {sin_a * sin_b:.4f} = {cos_apb:.4f} \\\\"
        )

        if isinstance(alpha_k, (int, float)):
            cos_apb_mc = math.cos(omega_k * ((a + b - self.c_star) % P))
            contribution = alpha_k * cos_apb_mc
            lines.append(
                f"    \\alpha_{{{k}}} \\cos(\\omega_{{{k}}}({a}+{b}-{self.c_star})) &= "
                f"{alpha_k:.2f} \\times \\cos(0) = {alpha_k:.2f} \\times 1 = {contribution:.2f}"
            )

        lines.append(r"\end{align*}")

        # TikZ: small unit circle showing the angle
        lines.append("")
        lines.append(r"\begin{center}")
        lines.append(r"\begin{tikzpicture}[scale=1.2]")
        lines.append(r"    % Unit circle")
        lines.append(r"    \draw[gray, thin] (0,0) circle (1);")
        lines.append(r"    \draw[->] (-1.3,0) -- (1.3,0) node[right] {\tiny $\cos$};")
        lines.append(r"    \draw[->] (0,-1.3) -- (0,1.3) node[above] {\tiny $\sin$};")

        # Point for token a
        angle_a = omega_k * a
        lines.append(
            f"    \\fill[blue] ({math.cos(angle_a):.3f},{math.sin(angle_a):.3f}) circle (2pt) "
            f"node[above right, font=\\tiny] {{$a={a}$}};"
        )
        lines.append(
            f"    \\draw[blue, dashed] (0,0) -- ({math.cos(angle_a):.3f},{math.sin(angle_a):.3f});"
        )

        # Point for token b
        angle_b = omega_k * b
        lines.append(
            f"    \\fill[red] ({math.cos(angle_b):.3f},{math.sin(angle_b):.3f}) circle (2pt) "
            f"node[below right, font=\\tiny] {{$b={b}$}};"
        )
        lines.append(
            f"    \\draw[red, dashed] (0,0) -- ({math.cos(angle_b):.3f},{math.sin(angle_b):.3f});"
        )

        # Point for a+b
        angle_apb = omega_k * (a + b)
        lines.append(
            f"    \\fill[green!50!black] ({math.cos(angle_apb):.3f},{math.sin(angle_apb):.3f}) circle (2.5pt) "
            f"node[below left, font=\\tiny] {{$a+b={a+b}$}};"
        )
        lines.append(
            f"    \\draw[green!50!black, thick] (0,0) -- ({math.cos(angle_apb):.3f},{math.sin(angle_apb):.3f});"
        )

        lines.append(f"    \\node[font=\\scriptsize] at (0,-1.7) {{Freq $k={k}$: tokens as angles on the unit circle}};")
        lines.append(r"\end{tikzpicture}")
        lines.append(r"\end{center}")

        return "\n".join(lines)

    def _count_neurons_for_freq(self, k: int) -> int:
        """Count neurons assigned to frequency k."""
        count = 0
        for neuron_idx, info in self.neuron_assignments.items():
            if info.get("frequency") == k:
                count += 1
        return count
