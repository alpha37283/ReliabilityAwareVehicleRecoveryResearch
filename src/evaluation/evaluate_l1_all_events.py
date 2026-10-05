import csv
from pathlib import Path
from evaluation.event_loader import load_event

from recovery.loss_detector import SimpleLossDetector


ROOT = Path(__file__).resolve().parents[2]

RESULTS = ROOT / "results/phase1"
REFERENCE_FILE = RESULTS / "phase1e_final_metrics.csv"
OUTPUT_FILE = RESULTS / "phase1f_l1_results.csv"

TRACKERS = ["sam21", "dam4sam"]


def as_bool(value):
    return str(value).strip().lower() == "true"


def load_reference():
    lookup = {}

    with REFERENCE_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            lookup[(row["event_id"], row["tracker"])] = row

    return lookup


def load_predictions(event_id, tracker):
    path = (
        RESULTS
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


def run_l1(event_id, tracker):
    event = load_event(event_id)

    occlusion_start = event.get("occlusion_start")
    reappearance = event.get("reappearance_frame")

    detector = SimpleLossDetector(
        lost_after=3,
        discontinuity_iou=0.10,
        discontinuity_jump=1.0,
    )

    for row in load_predictions(event_id, tracker):

        frame = row["frame_number"]

        # Evaluate L1 only inside the event's relevant loss interval.
        if occlusion_start is not None and frame < occlusion_start:
            continue

        if reappearance is not None and frame >= reappearance:
            break

        detector.update(
            frame_number=frame,
            present=row["present"],
            box=row["box"],
        )

    return detector.lost_frame


def main():
    reference = load_reference()

    rows = []

    tp = tn = fp = fn = 0
    delays = []

    for i in range(1, 13):
        event_id = f"E{i:03d}"

        for tracker in TRACKERS:

            ref = reference[(event_id, tracker)]

            ref_loss_text = ref["loss_frame"].strip()

            reference_loss_frame = (
                int(ref_loss_text)
                if ref_loss_text
                else None
            )

            reference_has_loss = (
                reference_loss_frame is not None
            )

            l1_loss_frame = run_l1(
                event_id,
                tracker,
            )

            l1_detected_loss = (
                l1_loss_frame is not None
            )

            if reference_has_loss and l1_detected_loss:
                classification = "TP"
                tp += 1

                delay = (
                    l1_loss_frame
                    - reference_loss_frame
                )

                delays.append(delay)

            elif (
                not reference_has_loss
                and not l1_detected_loss
            ):
                classification = "TN"
                tn += 1
                delay = None

            elif (
                not reference_has_loss
                and l1_detected_loss
            ):
                classification = "FP"
                fp += 1
                delay = None

            else:
                classification = "FN"
                fn += 1
                delay = None

            rows.append({
                "event_id": event_id,
                "tracker": tracker,
                "reference_loss_frame":
                    reference_loss_frame,
                "l1_loss_frame":
                    l1_loss_frame,
                "classification":
                    classification,
                "detection_delay_frames":
                    delay,
            })

    fields = [
        "event_id",
        "tracker",
        "reference_loss_frame",
        "l1_loss_frame",
        "classification",
        "detection_delay_frames",
    ]

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("PHASE 1F — L1 ALL-EVENT EVALUATION")
    print("=" * 50)
    print(f"Runs evaluated: {len(rows)}")
    print(f"True positives:  {tp}")
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")

    if delays:
        print(
            f"Mean detection delay on detected losses: "
            f"{sum(delays) / len(delays):.2f} frames"
        )

    print()
    print("MISMATCHES / IMPORTANT CASES")
    print("-" * 50)

    for row in rows:
        if (
            row["classification"] in {"FP", "FN"}
            or (
                row["classification"] == "TP"
                and row["detection_delay_frames"] != 0
            )
        ):
            print(
                f"{row['event_id']} | "
                f"{row['tracker']:7} | "
                f"reference={row['reference_loss_frame']} | "
                f"L1={row['l1_loss_frame']} | "
                f"{row['classification']} | "
                f"delay={row['detection_delay_frames']}"
            )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
