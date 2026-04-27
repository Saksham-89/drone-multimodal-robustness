from pathlib import Path
import numpy as np
from .base import BaseInferenceRunner


class EarlyFusionRunner(BaseInferenceRunner):
    """Inference wrapper for the early-fusion (concatenation) baseline.

    This model is implemented by us — RGB and TIR feature maps concatenated
    before the main encoder, ResNet-50 backbone.
    """

    def load_model(self, checkpoint_path: Path) -> None:
        raise NotImplementedError("Implement after defining the early fusion model architecture")

    def run(self, rgb: np.ndarray, tir: np.ndarray) -> list[dict]:
        raise NotImplementedError("Implement after defining the early fusion model architecture")
