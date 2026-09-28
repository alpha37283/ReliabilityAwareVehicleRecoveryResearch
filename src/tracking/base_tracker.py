from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """
    Common interface for all Phase-1 trackers.

    Every tracker must:
    1. initialize from the first frame + target box
    2. process later frames independently
    3. return predictions in a common format
    """

    @abstractmethod
    def initialize(self, frame, target_box):
        """
        Initialize the tracker.

        Parameters
        ----------
        frame:
            Image/frame used for initialization.

        target_box:
            Initial target box in xyxy format:
            [x1, y1, x2, y2]
        """
        pass

    @abstractmethod
    def track(self, frame):
        """
        Process one new frame.

        Returns
        -------
        dict with a common structure, for example:

        {
            "box": [x1, y1, x2, y2],
            "mask": None,
            "confidence": None,
            "present": True
        }

        Different trackers may provide more fields,
        but these common fields should remain available.
        """
        pass

    def reset(self):
        """
        Optional cleanup/reset between events.
        """
        pass