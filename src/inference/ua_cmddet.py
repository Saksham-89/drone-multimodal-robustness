from pathlib import Path
import numpy as np
from .base import BaseInferenceRunner


class UACMDetRunner(BaseInferenceRunner):
    """Inference wrapper for UA-CMDet (Sun et al., TCSVT 2022).

    Wire up after cloning the DroneVehicle repo into models/ua_cmddet/.
    """

    def load_model(self, checkpoint_path: Path) -> None:
        raise NotImplementedError("Wire up after cloning UA-CMDet repo")

    def run(self, rgb: np.ndarray, tir: np.ndarray) -> list[dict]:
        raise NotImplementedError("Wire up after cloning UA-CMDet repo")
