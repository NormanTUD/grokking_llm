class CircuitLatexGenerator:
    """
    Generates detailed LaTeX equations for a discovered Fourier multiplication circuit.
    Uses \\underbrace and \\overbrace extensively to show provenance of each term.
    """

    def __init__(self, P: int, key_frequencies: list[int], neuron_assignments: dict = None):
        self.P = P
        self.key_frequencies = key_frequencies
        self.neuron_assignments = neuron_assignments or {}

    def full_circuit_latex(self) -> str:
        """Generate the complete set of LaTeX equations for the discovered circuit."""
        sections = []

        sections.append(self._section_header())
        sections.append(self._embedding_equations())
        sections.append(self._attention_equations())
        sections.append(self._mlp_equations())
        sections.append(self._unembed_equations())
        sections.append(self._final_prediction())
        sections.append(self._constructive_interference())
        sections.append(self._worked_example())

        return "\n\n".join(sections)

    def _section_header(self) -> str:
        freqs_str = ", ".join(str(k) for k in self.key_frequencies)
        return (
            r"% =============================================================" + "\n"
            r"% FOURIER MULTIPLICATION CIRCUIT" + "\n"
            f"% P = {self.P}, key frequencies k \\in {{{freqs_str}}}" + "\n"
            r"% =============================================================" + "\n"
            r"\section{Discovered Circuit: Fourier Multiplication Algorithm}" + "\n\n"
            f"The model computes $(a + b) \\mod {self.P}$ using {len(self.key_frequencies)} "
            f"key frequencies $k \\in \\{{{freqs_str}\\}}$ with angular frequencies "
            r"$\omega_k = \frac{2\pi k}{" + str(self.P) + r"}$."
        )

    def _embedding_equations(self) -> str:
        P = self.P
        lines = []
        lines.append(r"\subsection{Step 1: Embedding (Token $\to$ Fourier Components)}")
        lines.append("")
        lines.append(r"The embedding matrix $W_E$ maps each one-hot token $t \in \{0, \ldots, "
                     f"{P-1}\\}}$ onto sinusoidal components:")
        lines.append("")
        lines.append(r"\begin{align}")

        # Show the embedding for token a
        lines.append(
            r"    \mathbf{x}^{(0)}_a &= "
            r"\underbrace{W_E \cdot \mathbf{e}_a}_{\text{token embedding of } a} "
            r"+ \underbrace{\mathbf{p}_0}_{\text{positional embedding (pos 0)}} \\"
        )
        lines.append(
            r"    &\approx \sum_{k \in \mathcal{K}} \bigg[ "
            r"\underbrace{\alpha_k \cos\!\left(\frac{2\pi k \cdot a}{" + str(P) + r"}\right)}"
            r"_{\substack{\text{cosine component} \\ \text{from } W_E \text{ row } a}} "
            r"\cdot \mathbf{u}_k^{(\cos)} "
            r"+ \underbrace{\beta_k \sin\!\left(\frac{2\pi k \cdot a}{" + str(P) + r"}\right)}"
            r"_{\substack{\text{sine component} \\ \text{from } W_E \text{ row } a}} "
            r"\cdot \mathbf{u}_k^{(\sin)} \bigg]"
        )
        lines.append(r"\end{align}")
        lines.append("")
        lines.append(r"where $\mathcal{K} = \{" +
                     ", ".join(str(k) for k in self.key_frequencies) +
                     r"\}$ are the key frequencies discovered via Fourier analysis of $W_E$.")
        lines.append("")
        lines.append(r"Similarly for token $b$ at position 1:")
        lines.append(r"\begin{align}")
        lines.append(
            r"    \mathbf{x}^{(0)}_b &= "
            r"\underbrace{W_E \cdot \mathbf{e}_b}_{\text{token embedding of } b} "
            r"+ \underbrace{\mathbf{p}_1}_{\text{positional embedding (pos 1)}}"
        )
        lines.append(r"\end{align}")

        return "\n".join(lines)

    def _attention_equations(self) -> str:
        P = self.P
        lines = []
        lines.append(r"\subsection{Step 2: Attention (Move Fourier Info to Output Position)}")
        lines.append("")
        lines.append(r"The attention heads compute scores from the `$=$' token (position 2) "
                     r"to positions 0 and 1, then produce a weighted combination:")
        lines.append("")
        lines.append(r"\begin{align}")

        # Attention score computation
        lines.append(
            r"    \text{score}^{(j)}_{=\to a} &= "
            r"\frac{"
            r"\overbrace{\mathbf{x}^{(0)}_= \cdot W_Q^{(j)}}^{\text{query from `=' position}}"
            r"\cdot "
            r"\overbrace{\left(W_K^{(j)}\right)^\top \!\cdot \mathbf{x}^{(0)}_a}^{\text{key from position } a}"
            r"}"
            r"{\underbrace{\sqrt{d_{\text{head}}}}_{\text{scaling factor}}} \\"
        )

        # Attention pattern (sigmoid form since softmax over 2 elements)
        lines.append(
            r"    A^{(j)}_0 &= \underbrace{\sigma\!\left("
            r"\text{score}^{(j)}_{=\to a} - \text{score}^{(j)}_{=\to b}"
            r"\right)}_{\substack{\text{softmax over 2 elements} \\ \text{= sigmoid of difference}}} \\"
        )

        # Show the periodic structure of attention
        lines.append(
            r"    &\approx \underbrace{0.5}_{\text{uniform baseline}} + "
            r"\underbrace{\gamma_j \Big("
            r"\cos\!\left(\omega_{k_j} \cdot a\right) - \cos\!\left(\omega_{k_j} \cdot b\right)"
            r"\Big)}_{\substack{\text{periodic modulation at frequency } k_j \\"
            r"\text{(from } C^{(j)} = W_E^\top W_K^{(j)\top} W_Q^{(j)} \mathbf{x}^{(0)}_= \text{)}}} \\"
        )

        # OV circuit output
        lines.append(
            r"    \text{attn\_out}^{(j)} &= "
            r"\underbrace{A^{(j)}_0}_{\text{attn weight to } a} \cdot "
            r"\overbrace{W_O^{(j)} W_V^{(j)} \mathbf{x}^{(0)}_a}^{\text{OV circuit applied to } a} "
            r"+ \underbrace{A^{(j)}_1}_{\text{attn weight to } b} \cdot "
            r"\overbrace{W_O^{(j)} W_V^{(j)} \mathbf{x}^{(0)}_b}^{\text{OV circuit applied to } b}"
        )

        lines.append(r"\end{align}")
        lines.append("")
        lines.append(
            r"\textbf{Key insight:} Since $A^{(j)}_0$ is approximately linear in "
            r"$\cos(\omega_{k_j} a)$ and the OV circuit outputs $\cos(\omega_{k_j} a)$, "
            r"their product creates \emph{degree-2 polynomials} of sines and cosines "
            r"--- exactly what is needed for the trig identity in the next step."
        )

        return "\n".join(lines)

    def _mlp_equations(self) -> str:
        P = self.P
        lines = []
        lines.append(r"\subsection{Step 3: MLP (Compute Trig Identities)}")
        lines.append("")
        lines.append(r"The residual stream after attention contains degree-2 products. "
                     r"The MLP neurons compute the key trigonometric identities:")
        lines.append("")
        lines.append(r"\begin{align}")

        # Residual mid
        lines.append(
            r"    \mathbf{x}^{(1)} &= "
            r"\underbrace{\mathbf{x}^{(0)}_=}_{\substack{\text{skip connection} \\ \text{(= token embedding)}}} "
            r"+ \sum_{j=0}^{3} "
            r"\underbrace{\text{attn\_out}^{(j)}}_{\substack{\text{attention head } j \\ "
            r"\text{output (degree-2 trig)}}} \\"
        )

        # MLP pre-activation
        lines.append(
            r"    \text{pre}_n &= "
            r"\underbrace{W_{\text{in}}[n, :] \cdot \mathbf{x}^{(1)}}_{\text{linear projection of neuron } n} "
            r"+ \underbrace{b_{\text{in}}[n]}_{\text{bias}} \\"
        )

        # ReLU
        lines.append(
            r"    \text{MLP}[n] &= "
            r"\underbrace{\text{ReLU}\!\left(\text{pre}_n\right)}_{\substack{"
            r"\text{activation of neuron } n \\ "
            r"\approx \text{ degree-2 polynomial of } \cos(\omega_k a), \sin(\omega_k a), \ldots}} \\"
        )

        lines.append(r"\end{align}")
        lines.append("")

        # The key trig identity
        lines.append(r"\textbf{The core computation:} For each key frequency $k$, "
                     r"the MLP neurons collectively compute:")
        lines.append("")
        lines.append(r"\begin{align}")
        lines.append(
            r"    \underbrace{\mathbf{u}_k^\top \cdot \text{MLP}(a,b)}_{"
            r"\substack{\text{projection of MLP activations} \\ "
            r"\text{onto direction } \mathbf{u}_k \text{ in } W_L}} "
            r"&\approx "
            r"\underbrace{\alpha_k \cos\!\left(\omega_k(a+b)\right)}_{"
            r"\text{target: cosine of sum}} \\"
        )
        lines.append(
            r"    &= \overbrace{"
            r"\underbrace{\alpha_k \cos(\omega_k a) \cos(\omega_k b)}_{"
            r"\substack{\text{from neurons computing} \\ \cos(\omega_k a)\cos(\omega_k b)}} "
            r"- \underbrace{\alpha_k \sin(\omega_k a) \sin(\omega_k b)}_{"
            r"\substack{\text{from neurons computing} \\ \sin(\omega_k a)\sin(\omega_k b)}}"
            r"}^{\text{cosine addition formula}} \\"
        )
        lines.append(r"    &\quad \notag \\")
        lines.append(
            r"    \underbrace{\mathbf{v}_k^\top \cdot \text{MLP}(a,b)}_{"
            r"\substack{\text{projection onto} \\ \text{direction } \mathbf{v}_k \text{ in } W_L}} "
            r"&\approx "
            r"\underbrace{\beta_k \sin\!\left(\omega_k(a+b)\right)}_{"
            r"\text{target: sine of sum}} \\"
        )
        lines.append(
            r"    &= \overbrace{"
            r"\underbrace{\beta_k \sin(\omega_k a) \cos(\omega_k b)}_{"
            r"\substack{\text{from neurons computing} \\ \sin(\omega_k a)\cos(\omega_k b)}} "
            r"+ \underbrace{\beta_k \cos(\omega_k a) \sin(\omega_k b)}_{"
            r"\substack{\text{from neurons computing} \\ \cos(\omega_k a)\sin(\omega_k b)}}"
            r"}^{\text{sine addition formula}}"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # Show where u_k and v_k come from
        lines.append(r"where $\mathbf{u}_k, \mathbf{v}_k \in \mathbb{R}^{512}$ are the "
                     r"directions in the neuron-logit map $W_L = W_{\text{out}}^\top W_U^\top$ "
                     r"corresponding to $\cos(\omega_k c)$ and $\sin(\omega_k c)$ respectively.")

        return "\n".join(lines)

    def _unembed_equations(self) -> str:
        P = self.P
        lines = []
        lines.append(r"\subsection{Step 4: Unembedding (Fourier $\to$ Logits)}")
        lines.append("")
        lines.append(r"The neuron-logit map $W_L = W_{\text{out}}^\top \cdot W_U^\top$ "
                     r"converts the MLP activations to output logits:")
        lines.append("")
        lines.append(r"\begin{align}")

        # W_L decomposition
        lines.append(
            r"    W_L &\approx \sum_{k \in \mathcal{K}} \bigg[ "
            r"\underbrace{\cos(\omega_k c)}_{\substack{\text{vector in } \mathbb{R}^P \\"
            r"\text{(c-th entry} = \cos(\omega_k c)\text{)}}} "
            r"\cdot \underbrace{\mathbf{u}_k^\top}_{\substack{\text{direction in neuron space} \\"
            r"\text{that reads } \cos(\omega_k(a+b))}} "
            r"+ \underbrace{\sin(\omega_k c)}_{\substack{\text{vector in } \mathbb{R}^P \\"
            r"\text{(c-th entry} = \sin(\omega_k c)\text{)}}} "
            r"\cdot \underbrace{\mathbf{v}_k^\top}_{\substack{\text{direction in neuron space} \\"
            r"\text{that reads } \sin(\omega_k(a+b))}} "
            r"\bigg] \\"
        )

        lines.append(r"\end{align}")
        lines.append("")

        # Logit computation
        lines.append(r"Therefore, the logit for output class $c$ is:")
        lines.append("")
        lines.append(r"\begin{align}")
        lines.append(
            r"    \text{Logit}(c \mid a, b) &= "
            r"\underbrace{W_L \cdot \text{MLP}(a,b)}_{\text{neuron-logit map applied to MLP output}} \\"
        )
        lines.append(
            r"    &\approx \sum_{k \in \mathcal{K}} \bigg[ "
            r"\underbrace{\cos(\omega_k c)}_{\substack{\text{from } W_L \\ \text{(unembed direction)}}} "
            r"\cdot \underbrace{\mathbf{u}_k^\top \text{MLP}(a,b)}_{"
            r"\substack{\text{MLP computes this} \\ \approx \alpha_k\cos(\omega_k(a+b))}} "
            r"+ \underbrace{\sin(\omega_k c)}_{\substack{\text{from } W_L \\ \text{(unembed direction)}}} "
            r"\cdot \underbrace{\mathbf{v}_k^\top \text{MLP}(a,b)}_{"
            r"\substack{\text{MLP computes this} \\ \approx \beta_k\sin(\omega_k(a+b))}} "
            r"\bigg] \\"
        )

        # Apply trig identity to simplify
        lines.append(
            r"    &\approx \sum_{k \in \mathcal{K}} "
            r"\underbrace{\alpha_k \Big["
            r"\cos(\omega_k c)\cos(\omega_k(a+b)) + \sin(\omega_k c)\sin(\omega_k(a+b))"
            r"\Big]}_{\text{cosine subtraction identity}} \\"
        )
        lines.append(
            r"    &= \boxed{\sum_{k \in \mathcal{K}} "
            r"\underbrace{\alpha_k \cos\!\left(\omega_k(a + b - c)\right)}_{"
            r"\substack{\text{peaks when } c \equiv a+b \pmod{" + str(P) + r"} \\"
            r"\text{since } \cos(0) = 1 \text{ is maximum}}}}"
        )
        lines.append(r"\end{align}")

        return "\n".join(lines)

    def _final_prediction(self) -> str:
        P = self.P
        lines = []
        lines.append(r"\subsection{Step 5: Prediction via Constructive Interference}")
        lines.append("")
        lines.append(r"\begin{align}")
        lines.append(
            r"    \hat{c} &= \underbrace{\arg\max_{c \in \{0,\ldots," + str(P-1) + r"\}}}_{"
            r"\text{select class with highest logit}} "
            r"\overbrace{\sum_{k \in \mathcal{K}} \alpha_k "
            r"\cos\!\left(\frac{2\pi k (a + b - c)}{" + str(P) + r"}\right)}^{"
            r"\text{sum of cosines at key frequencies}} \\"
        )
        lines.append(
            r"    &= \underbrace{(a + b) \mod " + str(P) + r"}_{"
            r"\substack{\text{constructive interference:} \\"
            r"\text{all cosines equal 1 when } c = a+b \bmod " + str(P) + r" \\"
            r"\text{destructive interference elsewhere}}}"
        )
        lines.append(r"\end{align}")

        return "\n".join(lines)

    def _constructive_interference(self) -> str:
        P = self.P
        lines = []
        lines.append(r"\subsection{Why Multiple Frequencies? (Constructive Interference)}")
        lines.append("")
        lines.append(r"A single cosine $\cos(\omega_k x)$ has period $" + str(P) +
                     r"$ but near-maxima at other values of $x$. "
                     r"By summing over multiple key frequencies, the model creates a function "
                     r"with a \emph{unique} maximum at $x = 0 \bmod " + str(P) + r"$:")
        lines.append("")
        lines.append(r"\begin{align}")
        lines.append(
            r"    f(x) &= \sum_{k \in \mathcal{K}} "
            r"\underbrace{\cos\!\left(\frac{2\pi k \cdot x}{" + str(P) + r"}\right)}_{"
            r"\text{each has max at } x=0} \\"
        )
        lines.append(
            r"    f(0) &= \underbrace{" + str(len(self.key_frequencies)) + r"}_{"
            r"\text{all } " + str(len(self.key_frequencies)) +
            r" \text{ cosines} = 1} \quad \gg \quad "
            r"f(x \neq 0) \quad "
            r"\underbrace{\text{(destructive interference)}}_{"
            r"\text{cosines at different frequencies cancel}}"
        )
        lines.append(r"\end{align}")

        return "\n".join(lines)

    def _worked_example(self) -> str:
        P = self.P
        a, b = 7, 13
        c_star = (a + b) % P
        lines = []
        lines.append(r"\subsection{Worked Example: $a=" + str(a) + r", b=" + str(b) + r"$}")
        lines.append("")
        lines.append(r"\noindent\textbf{Step 1: Embed.} "
                     r"Compute Fourier components for each key frequency $k$:")
        lines.append("")

        # Show embedding as a table-like structure
        lines.append(r"\begin{gather*}")
        for i, k in enumerate(self.key_frequencies[:3]):
            lines.append(
                f"    k={k}:\\quad"
                f" \\cos\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot {a}}}{{{P}}}\\right),\\;"
                f" \\sin\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot {a}}}{{{P}}}\\right),\\;"
                f" \\cos\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot {b}}}{{{P}}}\\right),\\;"
                f" \\sin\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot {b}}}{{{P}}}\\right)"
            )
            if i < 2:
                lines.append(r"    \\")
        lines.append(r"\end{gather*}")
        lines.append("")

        # MLP trig identity
        lines.append(r"\noindent\textbf{Step 2: MLP applies trig identity} "
                     r"$\cos(\alpha)\cos(\beta) - \sin(\alpha)\sin(\beta) = \cos(\alpha+\beta)$:")
        lines.append("")
        lines.append(r"\begin{gather*}")
        for i, k in enumerate(self.key_frequencies[:2]):
            lines.append(
                f"    k={k}:\\quad"
                f" \\underbrace{{"
                f"\\cos\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot {a}}}{{{P}}}\\right)"
                f"\\cos\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot {b}}}{{{P}}}\\right)"
                f" - "
                f"\\sin\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot {a}}}{{{P}}}\\right)"
                f"\\sin\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot {b}}}{{{P}}}\\right)"
                f"}}_{{= \\cos\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot {a+b}}}{{{P}}}\\right)}}"
            )
            if i < 1:
                lines.append(r"    \\[6pt]")
        lines.append(r"\end{gather*}")
        lines.append("")

        # Final logit
        lines.append(r"\noindent\textbf{Step 3: Logit for correct answer} "
                     f"$c^* = ({a}+{b}) \\bmod {P} = {c_star}$:")
        lines.append("")
        lines.append(r"\begin{equation*}")
        lines.append(
            r"    \Logit(" + str(c_star) + r") = \sum_{k \in \mathcal{K}} \alpha_k "
            r"\underbrace{\cos\!\left(\omega_k \cdot "
            r"\overbrace{(" + str(a) + "+" + str(b) + "-" + str(c_star) + r")}^{"
            r"= 0 \bmod " + str(P) + r"}\right)}_{= \cos(0) = 1}"
            r" = \sum_k \alpha_k \quad \text{(MAXIMUM)}"
        )
        lines.append(r"\end{equation*}")
        lines.append("")
        lines.append(
            r"For any $c \neq " + str(c_star) + r"$, the cosines at different frequencies "
            r"point in different directions and \textbf{destructively interfere}, "
            r"giving a smaller logit."
        )

        return "\n".join(lines)

    def per_frequency_detail(self, k: int) -> str:
        """Generate detailed LaTeX for a single key frequency k."""
        P = self.P
        lines = []
        lines.append(f"\\subsubsection{{Frequency $k = {k}$: "
                     f"$\\omega_{{{k}}} = \\frac{{2\\pi \\cdot {k}}}{{{P}}}$}}")
        lines.append("")
        lines.append(r"\noindent\textbf{Embedding:}")
        lines.append(r"\begin{align}")
        lines.append(
            f"    &\\underbrace{{W_E[a]}}_{{\\text{{row }} a \\text{{ of }} W_E}}"
            f" \\xrightarrow{{\\text{{DFT}}}}"
            f" \\underbrace{{\\cos\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot a}}{{{P}}}\\right)}}_"
            f"{{\\text{{component }} k={k}}},\\;"
            f"\\underbrace{{\\sin\\!\\left(\\tfrac{{2\\pi \\cdot {k} \\cdot a}}{{{P}}}\\right)}}_"
            f"{{\\text{{component }} k={k}}}"
            r"    \notag"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # Attention contribution
        lines.append(r"\noindent\textbf{Attention:}")
        lines.append(r"\begin{align}")
        lines.append(
            f"    &\\underbrace{{A^{{(j)}}_0 \\approx 0.5 + \\gamma_j\\cos(\\omega_{{{k}}} a)}}_"
            f"{{\\text{{head }} j \\text{{ tuned to freq }} {k}}}"
            f" \\times "
            f"\\underbrace{{\\text{{OV}}^{{(j)}}(a) \\approx \\cos(\\omega_{{{k}}} a)}}_"
            f"{{\\text{{OV output at freq }} {k}}}"
            r"    \notag \\"
        )
        lines.append(
            f"    &\\quad \\Rightarrow\\;"
            f"\\underbrace{{\\cos^2(\\omega_{{{k}}} a),\\;"
            f"\\cos(\\omega_{{{k}}} a)\\cos(\\omega_{{{k}}} b),\\; \\ldots}}_"
            f"{{\\text{{degree-2 products from attn}} \\times \\text{{OV}}}}"
            r"    \notag"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # MLP computation
        lines.append(r"\noindent\textbf{MLP (trig identity):}")
        lines.append(r"\begin{align}")
        lines.append(
            f"    &\\underbrace{{\\text{{neurons }} n \\in \\text{{cluster}}_{{{k}}}}}_"
            f"{{\\text{{{self._count_neurons_for_freq(k)} neurons at freq }} {k}}}"
            f" \\xrightarrow{{\\ReLU}}"
            r"    \notag \\"
        )
        lines.append(
            f"    &\\quad"
            f" \\overbrace{{"
            f"\\underbrace{{\\cos(\\omega_{{{k}}} a)\\cos(\\omega_{{{k}}} b)}}_"
            f"{{\\text{{from attn}}}}"
            f" - "
            f"\\underbrace{{\\sin(\\omega_{{{k}}} a)\\sin(\\omega_{{{k}}} b)}}_"
            f"{{\\text{{from attn}}}}"
            f"}}^{{= \\cos(\\omega_{{{k}}}(a+b))}}"
            r"    \notag"
        )
        lines.append(r"\end{align}")
        lines.append("")

        # Unembed readoff
        lines.append(r"\noindent\textbf{Unembedding:}")
        lines.append(r"\begin{align}")
        lines.append(
            f"    &\\underbrace{{\\bu_{{{k}}}^\\top \\cdot \\MLP}}_"
            f"{{\\text{{project in }} W_L}}"
            f" \\approx \\alpha_{{{k}}} \\cos(\\omega_{{{k}}}(a+b))"
            r"    \notag \\"
        )
        lines.append(
            f"    &\\quad \\xrightarrow{{\\times \\cos(\\omega_{{{k}}} c)}}"
            f" \\underbrace{{\\alpha_{{{k}}} \\cos(\\omega_{{{k}}}(a+b-c))}}_"
            f"{{\\text{{freq }} {k} \\text{{ contribution to logit}}(c)}}"
            r"    \notag"
        )
        lines.append(r"\end{align}")

        return "\n".join(lines)

    def _count_neurons_for_freq(self, k: int) -> int:
        """Count neurons assigned to frequency k."""
        count = 0
        for neuron_idx, info in self.neuron_assignments.items():
            if info.get("frequency") == k:
                count += 1
        return count


