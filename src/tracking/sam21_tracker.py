"""T0 SAM2.1 Tiny baseline adapter for Phase 1C.

This wraps Meta's unmodified SAM2.1 *video* predictor. The caller supplies an
ordered mapping of the actual UA-DETRAC frame numbers to JPEG paths, starting
with the selection frame and ending with the requested evaluation frame.

The predictor indexes its JPEG directory from zero, so this adapter makes
symlinks in a temporary directory, e.g. 00000.jpg -> img00309.jpg.
Only the selection box is used as a prompt; future ground truth is never read.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

from tracking.base_tracker import BaseTracker


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "external" / "sam2" / "checkpoints" / "sam2.1_hiera_tiny.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


class SAM21Tracker(BaseTracker):
    """Sequential SAM2.1 video predictions through our BaseTracker contract."""

    def __init__(self, video_frame_paths):
        if not video_frame_paths:
            raise ValueError("SAM2.1 needs event frame paths starting at selection.")
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"SAM2.1 checkpoint missing: {CHECKPOINT}")

        self.frame_numbers = sorted(video_frame_paths)
        self._next_index = 1  # frame index zero is used for initialization
        self._tmpdir = TemporaryDirectory(prefix="sam21_event_")
        self._video_dir = Path(self._tmpdir.name)
        self.state = None
        self._iterator = None
        self.predictor = None

        # Symlinks avoid copying/renaming the user's UA-DETRAC JPEGs.
        for idx, frame_number in enumerate(self.frame_numbers):
            src = Path(video_frame_paths[frame_number]).resolve()
            if not src.is_file() or src.suffix.lower() not in {".jpg", ".jpeg"}:
                raise FileNotFoundError(f"Expected JPEG for frame {frame_number}: {src}")
            (self._video_dir / f"{idx:05d}.jpg").symlink_to(src)

        self.predictor = build_sam2_video_predictor(
            CONFIG, str(CHECKPOINT), device="cuda"
        )

    @torch.inference_mode()
    def initialize(self, frame, target_box):
        if self.state is not None:
            raise RuntimeError("SAM2.1 adapter has already been initialized.")
        if len(target_box) != 4:
            raise ValueError("target_box must be xyxy: [x1,y1,x2,y2]")

        # All JPEGs are available as files, but the model only *tracks*
        # incrementally as next(self._iterator) is called by track().
        # Both offloads matter on a 4 GB Quadro P2000.
        self.state = self.predictor.init_state(
            video_path=str(self._video_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        if tuple(frame.size) != (
            self.state["video_width"], self.state["video_height"]
        ):
            raise ValueError("Selection frame size differs from the video frames.")

        self.predictor.add_new_points_or_box(
            inference_state=self.state,
            frame_idx=0,
            obj_id=1,
            box=np.asarray(target_box, dtype=np.float32),
        )
        self._iterator = self.predictor.propagate_in_video(
            self.state, start_frame_idx=1
        )

    @torch.inference_mode()
    def track(self, frame):
        # `frame` is accepted for compatibility with BaseTracker; SAM2 reads
        # the corresponding image from its indexed video state instead.
        if self._iterator is None:
            raise RuntimeError("Call initialize() before track().")
        if self._next_index >= len(self.frame_numbers):
            raise RuntimeError("There are no more event frames to track.")

        try:
            frame_idx, obj_ids, mask_logits = next(self._iterator)
        except StopIteration as exc:
            raise RuntimeError("SAM2.1 stopped before the event ended.") from exc

        expected_index = self._next_index
        if frame_idx != expected_index:
            raise RuntimeError(
                f"Frame order mismatch: expected index {expected_index}, got {frame_idx}"
            )
        self._next_index += 1
        obj_idx = list(obj_ids).index(1)
        mask = (mask_logits[obj_idx, 0] > 0).to("cpu").numpy().astype(np.uint8)
        ys, xs = np.nonzero(mask)

        if xs.size:
            # Use exclusive right/bottom bounds for xyxy box geometry.
            box = [
                float(xs.min()), float(ys.min()),
                float(xs.max() + 1), float(ys.max() + 1),
            ]
        else:
            box = None

        return {
            "box": box,
            "mask": mask,
            # Native SAM2.1 propagation does not expose a calibrated
            # frame-level target identity confidence through this API.
            "confidence": None,
            "present": box is not None,
        }

    @staticmethod
    def synchronize():
        """Called by the runner around timings for asynchronous CUDA work."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset(self):
        if self._iterator is not None:
            self._iterator.close()
            self._iterator = None
        if self.state is not None:
            self.predictor.reset_state(self.state)
            self.state = None
        self.predictor = None
        self._tmpdir.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
