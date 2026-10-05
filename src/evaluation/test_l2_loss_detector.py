import csv
from pathlib import Path

from recovery.l2_loss_detector import SentinelInspiredLossDetector


ROOT = Path(__file__).resolve().parents[2]

TESTS = [
    ("E006", "sam21"),
    ("E006", "dam4sam"),
    ("E012", "sam21"),
    ("E012", "dam4sam"),
]


def as_bool(x):
    return str(x).strip().lower() == "true"


def load_predictions(event_id, tracker):
    path = (
        ROOT
        / "results"
        / "phase1"
        / event_id
        / tracker
        / "predictions.csv"
    )

    rows = []

    with path.open(newline="") as f:
        for row in csv.DictReader(f):

            present = as_bool(row["present"])

            if present:
                box = [
                    float(row["box_x1"]),
                    float(row["box_y1"]),
                    float(row["box_x2"]),
                    float(row["box_y2"]),
                ]
            else:
                box = None

            rows.append({
                "frame_number": int(row["frame_number"]),
                "present": present,
                "box": box,
            })

    return rows


def fmt(x):
    return "-" if x is None else f"{x:.3f}"


def main():

    for event_id, tracker in TESTS:

        detector = SentinelInspiredLossDetector(
            iou_threshold=0.63,
            jump_threshold=0.12,
            stable_to_confident=3,
            critical_recovery_stable=3,
            anomaly_window=5,
            hard_anomalies_to_persistent=2,
            misses_to_lost=3,
        )

        previous_state = None
        previous_persistent = False

        print("\n" + "=" * 78)
        print(f"{event_id}/{tracker}")
        print("-" * 78)

        for row in load_predictions(event_id, tracker):

            result = detector.update(
                frame_number=row["frame_number"],
                present=row["present"],
                box=row["box"],
            )

            state = result["state"]
            persistent = result["persistent_critical"]

            state_changed = (
                state != previous_state
            )

            persistent_changed = (
                persistent != previous_persistent
            )

            if state_changed or persistent_changed:

                print(
                    f"frame={row['frame_number']} "
                    f"state={state:10} "
                    f"obs={result['observation']:14} "
                    f"IoU={fmt(result['temporal_iou'])} "
                    f"jump={fmt(result['center_jump'])} "
                    f"hard_window={result['hard_count_window']} "
                    f"persistent={persistent}"
                )

            previous_state = state
            previous_persistent = persistent

        print()
        print(f"FINAL STATE: {detector.state.value}")
        print(
            "PERSISTENT CRITICAL FRAME:",
            detector.persistent_critical_frame
        )
        print(
            "L2 LOST FRAME:",
            detector.lost_frame
        )


if __name__ == "__main__":
    main()
