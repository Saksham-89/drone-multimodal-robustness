from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class BaseInferenceRunner(ABC):
    """All three model runners implement this interface so experiment scripts are model-agnostic."""

    @abstractmethod
    def load_model(self, checkpoint_path: Path) -> None:
        """Load weights from checkpoint."""
        ...

    @abstractmethod
    def run(self, rgb: np.ndarray, tir: np.ndarray) -> list[dict]:
        """Run inference on a single (rgb, tir) pair.

        Returns list of detections: [{"bbox": [...], "score": float, "class": int}, ...]
        bbox format: oriented bounding box as 8 floats [x1,y1,x2,y2,x3,y3,x4,y4]
        """
        ...

    def run_rgb_only(self, rgb: np.ndarray, tir_shape: tuple) -> list[dict]:
        """Run with TIR zeroed (all zeros, same shape as real TIR)."""
        return self.run(rgb, np.zeros(tir_shape, dtype=np.uint8))

    def run_tir_only(self, tir: np.ndarray, rgb_shape: tuple) -> list[dict]:
        """Run with RGB zeroed (all zeros, same shape as real RGB)."""
        return self.run(np.zeros(rgb_shape, dtype=np.uint8), tir)
