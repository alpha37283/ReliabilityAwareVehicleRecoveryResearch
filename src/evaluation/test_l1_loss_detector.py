import csv
from pathlib import Path

from recovery.loss_detector import SimpleLossDetector


ROOT = Path(__file__).resolve().parents[2]

TESTS = [
    ("E006", "sam21"),
    ("E006", "dam4sam"),
    ("E012", "sam21"),
    ("E012", "dam4sam"),
]


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

            present = (
                row["present"].strip().lower()
                == "true"
            )

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
                "frame_number": int(
                    row["frame_number"]
                ),
                "present": present,
                "box": box,
            })

    return rows


def main():

    for event_id, tracker in TESTS:

        detector = SimpleLossDetector(
            lost_after=3,
            discontinuity_iou=0.10,
            discontinuity_jump=1.0,
        )

        rows = load_predictions(
            event_id,
            tracker,
        )

        previous_state = None

        print("\n" + "=" * 70)
        print(f"{event_id}/{tracker}")
        print("-" * 70)

        for row in rows:

            result = detector.update(
                frame_number=row["frame_number"],
                present=row["present"],
                box=row["box"],
            )

            state = result["state"]

            # Print only state transitions
            # or unreliable observations.
            if state != previous_state:
                print(
                    f"frame={row['frame_number']} "
                    f"state={state:9} "
                    f"reliable={result['reliable']} "
                    f"reason={result['reason']} "
                    f"streak={result['unreliable_streak']} "
                    f"lost_frame={result.get('lost_frame')}"
                )

            previous_state = state

        print()
        print(
            f"FINAL STATE: {detector.state.value}"
        )
        print(
            f"L1 LOST FRAME: {detector.lost_frame}"
        )


if __name__ == "__main__":
    main()
