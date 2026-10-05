from collections import deque
from enum import Enum

from recovery.loss_detector import (
    box_iou,
    normalized_center_jump,
)


class L2State(str, Enum):
    CONFIDENT = "CONFIDENT"
    UNCERTAIN = "UNCERTAIN"
    CRITICAL = "CRITICAL"
    LOST = "LOST"


class SentinelInspiredLossDetector:
    """
    L2: Sentinel-inspired reliability-state detector.

    Runtime signals:
        - prediction presence
        - temporal IoU
        - normalized center jump
        - recent anomaly history

    No ground truth is used.
    """

    def __init__(
        self,
        iou_threshold=0.63,
        jump_threshold=0.12,
        stable_to_confident=3,
        critical_recovery_stable=3,
        anomaly_window=5,
        hard_anomalies_to_persistent=2,
        misses_to_lost=3,
    ):
        self.iou_threshold = iou_threshold
        self.jump_threshold = jump_threshold

        self.stable_to_confident = stable_to_confident
        self.critical_recovery_stable = critical_recovery_stable

        self.anomaly_window = anomaly_window
        self.hard_anomalies_to_persistent = (
            hard_anomalies_to_persistent
        )

        self.misses_to_lost = misses_to_lost

        self.state = L2State.CONFIDENT

        self.previous_box = None

        self.stable_streak = 0
        self.miss_streak = 0

        self.history = deque(
            maxlen=self.anomaly_window
        )

        self.persistent_critical = False

        self.first_critical_frame = None
        self.persistent_critical_frame = None
        self.lost_frame = None


    @staticmethod
    def valid_box(box):
        if box is None:
            return False

        x1, y1, x2, y2 = box
        return x2 > x1 and y2 > y1


    def classify_observation(
        self,
        present,
        box,
    ):
        if not present or not self.valid_box(box):
            return "missing", None, None

        if self.previous_box is None:
            return "stable", None, None

        tiou = box_iou(
            self.previous_box,
            box,
        )

        jump = normalized_center_jump(
            self.previous_box,
            box,
        )

        low_iou = (
            tiou is not None
            and tiou < self.iou_threshold
        )

        high_jump = (
            jump is not None
            and jump > self.jump_threshold
        )

        if low_iou and high_jump:
            observation = "hard_anomaly"

        elif low_iou or high_jump:
            observation = "soft_anomaly"

        else:
            observation = "stable"

        return observation, tiou, jump


    def recent_hard_count(self):
        return sum(
            obs == "hard_anomaly"
            for obs in self.history
        )


    def enter_critical(self, frame_number):
        self.state = L2State.CRITICAL
        self.stable_streak = 0

        if self.first_critical_frame is None:
            self.first_critical_frame = frame_number


    def enter_lost(
        self,
        frame_number,
        start_frame=None,
    ):
        self.state = L2State.LOST

        if self.lost_frame is None:
            self.lost_frame = (
                start_frame
                if start_frame is not None
                else frame_number
            )


    def update(
        self,
        frame_number,
        present,
        box,
    ):
        if self.state == L2State.LOST:
            return {
                "frame_number": frame_number,
                "state": self.state.value,
                "observation": "already_lost",
                "temporal_iou": None,
                "center_jump": None,
                "stable_streak": self.stable_streak,
                "miss_streak": self.miss_streak,
                "hard_count_window":
                    self.recent_hard_count(),
                "persistent_critical":
                    self.persistent_critical,
                "persistent_critical_frame":
                    self.persistent_critical_frame,
                "lost_frame": self.lost_frame,
            }

        observation, tiou, jump = (
            self.classify_observation(
                present,
                box,
            )
        )

        self.history.append(observation)

        # --------------------------------------------------
        # Missing-output path
        # --------------------------------------------------

        if observation == "missing":

            self.stable_streak = 0
            self.miss_streak += 1

            if self.miss_streak == 1:
                if self.state != L2State.CRITICAL:
                    self.state = L2State.UNCERTAIN

            elif self.miss_streak == 2:
                self.enter_critical(
                    frame_number
                )

            elif self.miss_streak >= self.misses_to_lost:

                start_frame = (
                    frame_number
                    - self.misses_to_lost
                    + 1
                )

                self.enter_lost(
                    frame_number,
                    start_frame=start_frame,
                )

        # --------------------------------------------------
        # Valid prediction path
        # --------------------------------------------------

        else:

            self.miss_streak = 0

            # Repeated hard anomalies in recent history
            # create persistent CRITICAL status.
            if (
                self.recent_hard_count()
                >= self.hard_anomalies_to_persistent
            ):
                self.persistent_critical = True

                if self.persistent_critical_frame is None:
                    self.persistent_critical_frame = (
                        frame_number
                    )

                self.enter_critical(
                    frame_number
                )

            elif self.state == L2State.CONFIDENT:

                if observation == "stable":
                    self.stable_streak += 1

                elif observation == "soft_anomaly":
                    self.state = L2State.UNCERTAIN
                    self.stable_streak = 0

                elif observation == "hard_anomaly":
                    self.enter_critical(
                        frame_number
                    )

            elif self.state == L2State.UNCERTAIN:

                if observation == "stable":
                    self.stable_streak += 1

                    if (
                        self.stable_streak
                        >= self.stable_to_confident
                    ):
                        self.state = L2State.CONFIDENT
                        self.stable_streak = 0

                elif observation == "soft_anomaly":
                    self.stable_streak = 0

                elif observation == "hard_anomaly":
                    self.enter_critical(
                        frame_number
                    )

            elif self.state == L2State.CRITICAL:

                if self.persistent_critical:
                    # Smooth boxes alone cannot restore
                    # identity trust after repeated anomalies.
                    self.stable_streak = 0

                else:
                    if observation == "stable":
                        self.stable_streak += 1

                        if (
                            self.stable_streak
                            >= self.critical_recovery_stable
                        ):
                            # Recover cautiously:
                            # CRITICAL -> UNCERTAIN first.
                            self.state = L2State.UNCERTAIN
                            self.stable_streak = 0

                    elif observation == "soft_anomaly":
                        self.stable_streak = 0

                    elif observation == "hard_anomaly":
                        self.stable_streak = 0

        if present and self.valid_box(box):
            self.previous_box = list(box)

        return {
            "frame_number": frame_number,
            "state": self.state.value,
            "observation": observation,
            "temporal_iou": tiou,
            "center_jump": jump,
            "stable_streak": self.stable_streak,
            "miss_streak": self.miss_streak,
            "hard_count_window":
                self.recent_hard_count(),
            "persistent_critical":
                self.persistent_critical,
            "persistent_critical_frame":
                self.persistent_critical_frame,
            "lost_frame": self.lost_frame,
        }
