from dataclasses import dataclass, field
import numpy as np

@dataclass
class DiscoveredCircuit:
    """Stores the results of Fourier-based circuit discovery."""
    key_frequencies: list = field(default_factory=list)
    embedding_fourier_norms: np.ndarray = field(default_factory=lambda: np.array([]))
    wl_fourier_norms: np.ndarray = field(default_factory=lambda: np.array([]))
    neuron_frequency_assignments: dict = field(default_factory=dict)
    fve_logits: float = 0.0
    verification_accuracy: float = 0.0
    mathematical_formula: str = ""
    algorithm_description: str = ""
