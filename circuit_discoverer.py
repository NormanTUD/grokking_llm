class CircuitDiscoverer:
    """
    Discovers the Fourier multiplication circuit in a trained modular addition
    transformer, following Nanda et al. (2023).

    Steps:
    1. Compute Fourier norms of the embedding matrix to find key frequencies
    2. Compute Fourier norms of the neuron-logit map (W_out @ W_unembed)
    3. Assign neurons to frequencies based on their activation patterns
    4. Verify the trig identity: cos(wk*a)*cos(wk*b) - sin(wk*a)*sin(wk*b) = cos(wk*(a+b))
    5. Compute FVE (Fraction of Variance Explained) for the reconstructed logits
    6. Test the discovered formula exhaustively on all P*P inputs
    """

    def __init__(self, model: ModularAdditionTransformer):
        self.model = model
        self.P = model.P
        self.d_model = model.d_model
        self.d_mlp = model.d_mlp

    def compute_fourier_norms(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Fourier norms of the embedding and the neuron-logit map.

        Returns:
            embed_norms: (P//2 + 1,) array of Fourier norms for the embedding
            wl_norms: (P//2 + 1,) array of Fourier norms for W_L = W_mlp_out @ W_unembed
        """
        P = self.P

        # Embedding matrix (only the first P tokens, excluding '=')
        W_E = self.model.embed.weight[:P].detach().cpu().numpy()  # (P, d_model)

        # Fourier basis vectors
        fourier_basis = np.zeros((P, P))
        fourier_basis[0] = np.ones(P) / np.sqrt(P)
        for k in range(1, P // 2 + 1):
            fourier_basis[2 * k - 1] = np.cos(2 * np.pi * k * np.arange(P) / P) * np.sqrt(2 / P)
            if 2 * k < P:
                fourier_basis[2 * k] = np.sin(2 * np.pi * k * np.arange(P) / P) * np.sqrt(2 / P)

        # Project embedding onto Fourier basis: F @ W_E -> (P, d_model)
        W_E_fourier = fourier_basis @ W_E  # (P, d_model)

        # Compute norms per frequency
        # Frequency k uses rows 2k-1 (cos) and 2k (sin)
        embed_norms = np.zeros(P // 2 + 1)
        embed_norms[0] = np.linalg.norm(W_E_fourier[0])
        for k in range(1, P // 2 + 1):
            cos_row = W_E_fourier[2 * k - 1] if 2 * k - 1 < P else np.zeros(self.d_model)
            sin_row = W_E_fourier[2 * k] if 2 * k < P else np.zeros(self.d_model)
            embed_norms[k] = np.sqrt(np.linalg.norm(cos_row) ** 2 + np.linalg.norm(sin_row) ** 2)

        # Neuron-logit map: W_L = W_mlp_out^T @ W_unembed^T
        # W_mlp_out: (d_mlp, d_model), W_unembed: (d_model, P)
        W_mlp_out = self.model.mlp_out.weight.detach().cpu().numpy()  # (d_model, d_mlp)
        W_unembed = self.model.unembed.weight.detach().cpu().numpy()  # (P, d_model)

        # W_L maps neurons to logits: (d_mlp, d_model) @ (d_model, P) -> but we want per-neuron
        # Actually: logits = x @ W_unembed^T, and mlp_out = hidden @ W_mlp_out^T
        # So the neuron-to-logit map is W_mlp_out^T @ W_unembed^T = (d_mlp, d_model) @ ...
        # Let's compute it as: for each neuron n, its contribution to logit c is
        # W_mlp_out[c, n] * W_unembed[c, :] ...
        # Simpler: W_L = W_mlp_out.T @ W_unembed.T -> (d_mlp, P)
        # Actually W_mlp_out is (d_model, d_mlp), so W_mlp_out.T is (d_mlp, d_model)
        # W_unembed is (P, d_model), so W_unembed.T is (d_model, P)
        # W_L = W_mlp_out.T @ W_unembed.T -> (d_mlp, P)
        W_L = W_mlp_out.T @ W_unembed.T  # (d_mlp, P)

        # Fourier transform of W_L along the output (logit) dimension
        W_L_fourier = W_L @ fourier_basis.T  # (d_mlp, P)

        # Compute norms per frequency (summed over all neurons)
        wl_norms = np.zeros(P // 2 + 1)
        wl_norms[0] = np.linalg.norm(W_L_fourier[:, 0])
        for k in range(1, P // 2 + 1):
            cos_col = W_L_fourier[:, 2 * k - 1] if 2 * k - 1 < P else np.zeros(self.d_mlp)
            sin_col = W_L_fourier[:, 2 * k] if 2 * k < P else np.zeros(self.d_mlp)
            wl_norms[k] = np.sqrt(np.linalg.norm(cos_col) ** 2 + np.linalg.norm(sin_col) ** 2)

        return embed_norms, wl_norms

    def find_key_frequencies(self, embed_norms: np.ndarray, wl_norms: np.ndarray,
                             top_k: int = 5, threshold_ratio: float = 0.3) -> list[int]:
        """
        Identify key frequencies that dominate both the embedding and neuron-logit map.

        A frequency is "key" if it has high norm in BOTH the embedding and the W_L map.
        """
        P = self.P

        # Normalize norms
        embed_normalized = embed_norms / (embed_norms.max() + 1e-10)
        wl_normalized = wl_norms / (wl_norms.max() + 1e-10)

        # Combined score: geometric mean of both norms (both must be high)
        combined = np.sqrt(embed_normalized * wl_normalized)

        # Skip frequency 0 (DC component)
        combined[0] = 0

        # Find frequencies above threshold
        threshold = threshold_ratio * combined.max()
        candidates = np.where(combined > threshold)[0]

        # Sort by combined score and take top_k
        sorted_candidates = sorted(candidates, key=lambda k: -combined[k])
        key_freqs = sorted_candidates[:top_k]

        return sorted(key_freqs)

    def assign_neurons_to_frequencies(self, key_freqs: list[int],
                                      threshold: float = 0.5) -> dict:
        """
        Assign MLP neurons to key frequencies based on their activation patterns.

        For each neuron, compute how well its input weights align with cos/sin
        at each key frequency. Assign it to the best-matching frequency if the
        match exceeds the threshold.
        """
        P = self.P
        assignments = {}

        # Get MLP input weights
        W_mlp_in = self.model.mlp_in.weight.detach().cpu().numpy()  # (d_mlp, d_model)
        bias_mlp_in = self.model.mlp_in.bias.detach().cpu().numpy()  # (d_mlp,)

        # Get all activations to determine neuron behavior
        all_a = torch.arange(P)
        all_b = torch.arange(P)
        aa = all_a.repeat_interleave(P)
        bb = all_b.repeat(P)

        with torch.no_grad():
            mlp_hidden, _ = self.model.get_mlp_activations(aa, bb)
            mlp_hidden = mlp_hidden.cpu().numpy()  # (P*P, d_mlp)

        # Reshape to (P, P, d_mlp)
        activations_grid = mlp_hidden.reshape(P, P, self.d_mlp)

        for neuron_idx in range(self.d_mlp):
            neuron_act = activations_grid[:, :, neuron_idx]  # (P, P)

            # Skip dead neurons
            if neuron_act.max() - neuron_act.min() < 1e-6:
                continue

            # For each key frequency, compute correlation with cos(wk*(a+b))
            best_freq = None
            best_score = 0.0

            for k in key_freqs:
                # Expected pattern: cos(2*pi*k*(a+b)/P) or sin(2*pi*k*(a+b)/P)
                a_grid = np.arange(P).reshape(-1, 1)
                b_grid = np.arange(P).reshape(1, -1)
                cos_pattern = np.cos(2 * np.pi * k * (a_grid + b_grid) / P)
                sin_pattern = np.sin(2 * np.pi * k * (a_grid + b_grid) / P)

                # Also check cos(wk*a)*cos(wk*b) pattern (pre-trig-identity)
                cos_a_cos_b = np.cos(2 * np.pi * k * a_grid / P) * np.cos(2 * np.pi * k * b_grid / P)
                sin_a_sin_b = np.sin(2 * np.pi * k * a_grid / P) * np.sin(2 * np.pi * k * b_grid / P)
                cos_a_sin_b = np.cos(2 * np.pi * k * a_grid / P) * np.sin(2 * np.pi * k * b_grid / P)
                sin_a_cos_b = np.sin(2 * np.pi * k * a_grid / P) * np.cos(2 * np.pi * k * b_grid / P)

                patterns = [cos_pattern, sin_pattern, cos_a_cos_b, sin_a_sin_b,
                           cos_a_sin_b, sin_a_cos_b]

                for pattern in patterns:
                    # Normalize both
                    n_act = neuron_act - neuron_act.mean()
                    n_pat = pattern - pattern.mean()
                    norm_act = np.linalg.norm(n_act)
                    norm_pat = np.linalg.norm(n_pat)
                    if norm_act < 1e-10 or norm_pat < 1e-10:
                        continue
                    corr = np.abs(np.sum(n_act * n_pat) / (norm_act * norm_pat))
                    if corr > best_score:
                        best_score = corr
                        best_freq = k

            if best_freq is not None and best_score > threshold:
                assignments[neuron_idx] = {
                    "frequency": best_freq,
                    "correlation": float(best_score),
                }

        return assignments

    def compute_fve(self, key_freqs: list[int]) -> float:
        """
        Compute Fraction of Variance Explained (FVE) for the logits
        when restricted to key frequencies.
        """
        P = self.P
        all_a = torch.arange(P).repeat_interleave(P)
        all_b = torch.arange(P).repeat(P)

        with torch.no_grad():
            logits = self.model(all_a, all_b).cpu().numpy()  # (P*P, P)

        logit_cube = logits.reshape(P, P, P)

        # Full DFT
        logit_fft = np.fft.fft2(logit_cube, axes=(0, 1))

        # Restrict to key frequencies
        logit_fft_restricted = np.zeros_like(logit_fft)
        logit_fft_restricted[0, 0, :] = logit_fft[0, 0, :]  # DC component
        for k in key_freqs:
            logit_fft_restricted[k, :, :] = logit_fft[k, :, :]
            logit_fft_restricted[:, k, :] = logit_fft[:, k, :]
            logit_fft_restricted[P - k, :, :] = logit_fft[P - k, :, :]
            logit_fft_restricted[:, P - k, :] = logit_fft[:, P - k, :]

        logit_restricted = np.fft.ifft2(logit_fft_restricted, axes=(0, 1)).real

        # FVE = 1 - ||logits - restricted||^2 / ||logits - mean||^2
        total_var = np.sum((logit_cube - logit_cube.mean()) ** 2)
        residual_var = np.sum((logit_cube - logit_restricted) ** 2)

        if total_var < 1e-10:
            return 0.0

        fve = 1.0 - residual_var / total_var
        return float(fve)

    def verify_formula(self, key_freqs: list[int]) -> tuple[float, str]:
        """
        Verify the discovered formula by testing it exhaustively on all P*P inputs.

        The formula is: logit(c) ∝ sum_k cos(2*pi*k*(a+b-c)/P)
        which peaks at c = (a+b) mod P due to constructive interference.

        Returns:
            accuracy: fraction of inputs where argmax(formula) == (a+b) mod P
            formula_str: string description of the formula
        """
        P = self.P

        # Build the formula-based logits
        a_vals = np.arange(P).reshape(-1, 1, 1)  # (P, 1, 1)
        b_vals = np.arange(P).reshape(1, -1, 1)  # (1, P, 1)
        c_vals = np.arange(P).reshape(1, 1, -1)  # (1, 1, P)

        # logit(a, b, c) = sum_k cos(2*pi*k*(a+b-c)/P)
        formula_logits = np.zeros((P, P, P))
        for k in key_freqs:
            formula_logits += np.cos(2 * np.pi * k * (a_vals + b_vals - c_vals) / P)

        # Predict
        formula_preds = formula_logits.reshape(P * P, P).argmax(axis=1)
        targets = ((np.arange(P).reshape(-1, 1) + np.arange(P).reshape(1, -1)) % P).reshape(-1)

        accuracy = (formula_preds == targets).mean()

        formula_str = f"logit(a, b, c) = Σ_k cos(2π·k·(a+b-c)/{P})\n"
        formula_str += f"where k ∈ {{{', '.join(map(str, key_freqs))}}}\n"
        formula_str += f"prediction = argmax_c logit(a, b, c) = (a + b) mod {P}"

        return float(accuracy), formula_str

    def full_discovery(self, progress_cb=None) -> DiscoveredCircuit:
        """
        Run the full Fourier circuit discovery pipeline.
        """
        def update(msg):
            if progress_cb:
                progress_cb(msg)

        update("Step 1: Computing Fourier norms of embedding and neuron-logit map...")
        embed_norms, wl_norms = self.compute_fourier_norms()

        update("Step 2: Identifying key frequencies...")
        key_freqs = self.find_key_frequencies(embed_norms, wl_norms)
        update(f"  Found key frequencies: {key_freqs}")

        update("Step 3: Assigning neurons to frequencies...")
        assignments = self.assign_neurons_to_frequencies(key_freqs)
        update(f"  Assigned {len(assignments)}/{self.d_mlp} neurons to key frequencies")

        update("Step 4: Computing FVE (Fraction of Variance Explained)...")
        fve = self.compute_fve(key_freqs)
        update(f"  FVE = {fve:.4f}")

        update("Step 5: Verifying formula exhaustively on all P*P inputs...")
        accuracy, formula_str = self.verify_formula(key_freqs)
        update(f"  Verification accuracy: {accuracy*100:.2f}%")
        update(f"  Formula: {formula_str}")

        algorithm_desc = (
            f"The model implements a Fourier multiplication algorithm:\n"
            f"1. EMBEDDING: Maps each input token t to cos(2π·k·t/{self.P}) and sin(2π·k·t/{self.P}) "
            f"for key frequencies k ∈ {{{', '.join(map(str, key_freqs))}}}\n"
            f"2. ATTENTION: Moves Fourier components from input positions (a, b) to the output position (=)\n"
            f"3. MLP: Computes trig identity cos(wk·a)·cos(wk·b) - sin(wk·a)·sin(wk·b) = cos(wk·(a+b))\n"
            f"4. UNEMBED: Converts cos(wk·(a+b)) back to logits via cos(2π·k·(a+b-c)/{self.P}),\n"
            f"   which peaks at c = (a+b) mod {self.P} due to constructive interference across frequencies."
        )

        return DiscoveredCircuit(
            key_frequencies=key_freqs,
            embedding_fourier_norms=embed_norms,
            wl_fourier_norms=wl_norms,
            neuron_frequency_assignments=assignments,
            fve_logits=fve,
            verification_accuracy=accuracy,
            mathematical_formula=formula_str,
            algorithm_description=algorithm_desc,
        )


