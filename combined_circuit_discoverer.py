from modular_addition_transformer import ModularAdditionTransformer
from discovered_circuit import DiscoveredCircuit
from circuit_discoverer import CircuitDiscoverer
from acc_circuit_discoverer import ACDCCircuitDiscoverer

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

