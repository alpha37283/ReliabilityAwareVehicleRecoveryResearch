import math
from enum import Enum


class LossState(str, Enum):
    TRACKING = "TRACKING"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"


def box_iou(a, b):
    if a is None or b is None:
        return None

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def normalized_center_jump(previous_box, current_box):
    if previous_box is None or current_box is None:
        return None

    px1, py1, px2, py2 = previous_box
    x1, y1, x2, y2 = current_box

    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    pw = px2 - px1
    ph = py2 - py1

    previous_diagonal = math.sqrt(
        pw * pw + ph * ph
    )

    if previous_diagonal <= 0:
        return None

    jump = math.sqrt(
        (cx - pcx) ** 2 +
        (cy - pcy) ** 2
    )

    return jump / previous_diagonal


class SimpleLossDetector:
    """
    L1: simple online loss detector.

    Uses runtime tracker outputs only.
    No GT boxes, target IDs or annotation IoU are used.
    """

    def __init__(
        self,
        lost_after=3,
        discontinuity_iou=0.10,
        discontinuity_jump=1.0,
    ):
        self.lost_after = lost_after
        self.discontinuity_iou = discontinuity_iou
        self.discontinuity_jump = discontinuity_jump

        self.state = LossState.TRACKING
        self.unreliable_streak = 0

        self.previous_box = None
        self.lost_frame = None

    @staticmethod
    def valid_box(box):
        if box is None:
            return False

        x1, y1, x2, y2 = box

        return x2 > x1 and y2 > y1

    def update(
        self,
        frame_number,
        present,
        box,
    ):
        # LOST is deliberately sticky.
        # Recovery must be handled by later recovery logic.
        if self.state == LossState.LOST:
            return {
                "frame_number": frame_number,
                "state": self.state.value,
                "reliable": False,
                "reason": "already_lost",
                "unreliable_streak": self.unreliable_streak,
            }

        reason = "reliable"
        reliable = True

        if not present or not self.valid_box(box):
            reliable = False
            reason = "missing_prediction"

        elif self.previous_box is not None:
            tiou = box_iou(
                self.previous_box,
                box,
            )

            jump = normalized_center_jump(
                self.previous_box,
                box,
            )

            if (
                tiou is not None
                and jump is not None
                and tiou < self.discontinuity_iou
                and jump > self.discontinuity_jump
            ):
                reliable = False
                reason = "extreme_temporal_discontinuity"

        if reliable:
            self.unreliable_streak = 0
            self.state = LossState.TRACKING

        else:
            self.unreliable_streak += 1

            if self.unreliable_streak >= self.lost_after:
                self.state = LossState.LOST

                if self.lost_frame is None:
                    self.lost_frame = (
                        frame_number
                        - self.lost_after
                        + 1
                    )
            else:
                self.state = LossState.UNCERTAIN

        # Only reliable observations update trusted history.
        if reliable:
            self.previous_box = list(box)

        return {
            "frame_number": frame_number,
            "state": self.state.value,
            "reliable": reliable,
            "reason": reason,
            "unreliable_streak": self.unreliable_streak,
            "lost_frame": self.lost_frame,
        }
