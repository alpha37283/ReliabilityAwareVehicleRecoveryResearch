from tracking.base_tracker import BaseTracker


class DebugStaticTracker(BaseTracker):
    """
    Temporary infrastructure-only tracker.

    It simply returns the initialization box for every frame.
    It is NOT a research baseline.
    """

    def __init__(self):
        self.target_box = None

    def initialize(self, frame, target_box):
        self.target_box = list(target_box)

    def track(self, frame):
        if self.target_box is None:
            raise RuntimeError(
                "Tracker must be initialized first."
            )

        return {
            "box": self.target_box.copy(),
            "mask": None,
            "confidence": None,
            "present": True,
        }

    def reset(self):
        self.target_box = None