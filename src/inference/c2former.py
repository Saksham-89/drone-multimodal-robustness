from pathlib import Path
import numpy as np
from .base import BaseInferenceRunner


class C2FormerRunner(BaseInferenceRunner):
    """Inference wrapper for C2Former (Yuan & Wei, IEEE TGRS 2024).

    Wire up after cloning the C2Former repo into models/c2former/.
    """

    def load_model(self, checkpoint_path: Path) -> None:
        raise NotImplementedError("Wire up after cloning C2Former repo")

    def run(self, rgb: np.ndarray, tir: np.ndarray) -> list[dict]:
        raise NotImplementedError("Wire up after cloning C2Former repo")
