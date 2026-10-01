"""Phase 1C T1 adapter: original DAM4SAM (SAM2.1 Tiny backbone).

Keep the authors' implementation in external/DAM4SAM unchanged.
Use the same initial selection box and output contract as T0 SAM2.1.
Each experiment launches in the ``vehicle_dam4sam`` conda environment.
"""

import gc
import sys
from pathlib import Path

import numpy as np
import torch

from tracking.base_tracker import BaseTracker


ROOT = Path(__file__).resolve().parents[2]
DAM_REPO = ROOT / "external" / "DAM4SAM"
TINY_CKPT = DAM_REPO / "sam2.1_hiera_tiny.pt"
TINY_CONFIG = DAM_REPO / "sam2" / "sam21pp_hiera_t.yaml"


class DAM4SAMAdapter(BaseTracker):
    """Wrap the unmodified authors' DAM4SAMTracker as a one-target tracker."""

    def __init__(self):
        if not DAM_REPO.is_dir():
            raise FileNotFoundError(f"Missing DAM4SAM repository: {DAM_REPO}")
        if not TINY_CKPT.is_file():
            raise FileNotFoundError(f"Missing DAM4SAM Tiny checkpoint: {TINY_CKPT}")
        if not TINY_CONFIG.is_file():
            raise FileNotFoundError(f"Missing DAM4SAM Tiny config: {TINY_CONFIG}")

        # DAM4SAM ships a modified `sam2` and imports `utils` from its root.
        # This adapter is run in its own conda environment, so it must NOT
        # import Meta's separately installed plain SAM2 implementation.
        dam_path = str(DAM_REPO)
        if dam_path not in sys.path:
            sys.path.insert(0, dam_path)

        from dam4sam_tracker import DAM4SAMTracker

        self._impl = DAM4SAMTracker(tracker_name="sam21pp-T")
        self._initialized = False

    @torch.inference_mode()
    def initialize(self, frame, target_box):
        """Use only the selected frame and annotated xyxy prompt."""
        if self._initialized:
            raise RuntimeError("DAM4SAM adapter already initialized.")
        if len(target_box) != 4:
            raise ValueError("target_box must be [x1, y1, x2, y2]")

        x1, y1, x2, y2 = [float(value) for value in target_box]
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid xyxy target box: {target_box}")

        # Authors' initialize(image, init_mask, bbox) takes bbox = xywh,
        # NOT the xyxy coordinates used by our Phase 1B runner.
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
        self._impl.initialize(frame, init_mask=None, bbox=bbox_xywh)
        self._initialized = True

    @torch.inference_mode()
    def track(self, frame):
        if not self._initialized:
            raise RuntimeError("Call initialize() before track().")

        outputs = self._impl.track(frame)
        if not isinstance(outputs, dict) or "pred_mask" not in outputs:
            raise RuntimeError("DAM4SAM did not return a pred_mask.")

        mask = np.asarray(outputs["pred_mask"])
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D predicted mask; got {mask.shape}")
        mask = (mask > 0).astype(np.uint8)

        ys, xs = np.nonzero(mask)
        if xs.size:
            # xyxy with exclusive right/bottom edges, matching T0.
            box = [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ]
        else:
            box = None

        return {
            "box": box,
            "mask": mask,
            # DAM4SAM's public track() returns only the mask, not a
            # calibrated probability of the target's identity.
            "confidence": None,
            "present": box is not None,
        }

    @staticmethod
    def synchronize():
        """Make existing runner's GPU timing accurate for async kernels."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset(self):
        if self._impl is not None:
            try:
                inference_state = getattr(self._impl, "inference_state", None)
                if inference_state is not None:
                    self._impl.predictor.reset_state(inference_state)
            finally:
                self._impl = None
                self._initialized = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
