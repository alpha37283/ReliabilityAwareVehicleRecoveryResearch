import csv
from pathlib import Path

from evaluation.event_loader import load_event
from recovery.l2_loss_detector import SentinelInspiredLossDetector


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/phase1"

REFERENCE_FILE = RESULTS / "phase1e_final_metrics.csv"
OUTPUT_FILE = RESULTS / "phase1f_l2_results.csv"

TRACKERS = ["sam21", "dam4sam"]


def as_bool(value):
    return str(value).strip().lower() == "true"


def load_reference():
    out = {}

    with REFERENCE_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["event_id"], row["tracker"])] = row

    return out


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


def run_l2(event_id, tracker):
    event = load_event(event_id)

    occlusion_start = event.get("occlusion_start")
    reappearance = event.get("reappearance_frame")

    detector = SentinelInspiredLossDetector(
        iou_threshold=0.63,
        jump_threshold=0.12,
        stable_to_confident=3,
        critical_recovery_stable=3,
        anomaly_window=5,
        hard_anomalies_to_persistent=2,
        misses_to_lost=3,
    )

    for row in load_predictions(event_id, tracker):

        frame = row["frame_number"]

        if occlusion_start is not None and frame < occlusion_start:
            continue

        if reappearance is not None and frame >= reappearance:
            break

        detector.update(
            frame_number=frame,
            present=row["present"],
            box=row["box"],
        )

    return (
        detector.persistent_critical_frame,
        detector.lost_frame,
        detector.state.value,
    )


def classification(reference_has_loss, detected):
    if reference_has_loss and detected:
        return "TP"

    if not reference_has_loss and not detected:
        return "TN"

    if not reference_has_loss and detected:
        return "FP"

    return "FN"


def main():
    reference = load_reference()

    rows = []

    risk_counts = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }

    lost_counts = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }

    risk_delays = []
    lost_delays = []

    for i in range(1, 13):

        event_id = f"E{i:03d}"

        for tracker in TRACKERS:

            ref = reference[(event_id, tracker)]

            ref_text = ref["loss_frame"].strip()

            ref_loss = (
                int(ref_text)
                if ref_text
                else None
            )

            reference_has_loss = (
                ref_loss is not None
            )

            persistent_frame, lost_frame, final_state = (
                run_l2(
                    event_id,
                    tracker,
                )
            )

            # Recovery/risk trigger:
            # persistent CRITICAL OR explicit LOST.
            trigger_candidates = [
                f
                for f in [
                    persistent_frame,
                    lost_frame,
                ]
                if f is not None
            ]

            risk_frame = (
                min(trigger_candidates)
                if trigger_candidates
                else None
            )

            risk_detected = (
                risk_frame is not None
            )

            lost_detected = (
                lost_frame is not None
            )

            risk_class = classification(
                reference_has_loss,
                risk_detected,
            )

            lost_class = classification(
                reference_has_loss,
                lost_detected,
            )

            risk_counts[risk_class] += 1
            lost_counts[lost_class] += 1

            risk_delay = None
            lost_delay = None

            if reference_has_loss and risk_detected:
                risk_delay = (
                    risk_frame - ref_loss
                )
                risk_delays.append(risk_delay)

            if reference_has_loss and lost_detected:
                lost_delay = (
                    lost_frame - ref_loss
                )
                lost_delays.append(lost_delay)

            rows.append({
                "event_id": event_id,
                "tracker": tracker,
                "reference_loss_frame": ref_loss,
                "persistent_critical_frame":
                    persistent_frame,
                "lost_frame": lost_frame,
                "risk_trigger_frame": risk_frame,
                "risk_classification":
                    risk_class,
                "risk_delay_frames":
                    risk_delay,
                "lost_classification":
                    lost_class,
                "lost_delay_frames":
                    lost_delay,
                "final_state":
                    final_state,
            })

    fields = [
        "event_id",
        "tracker",
        "reference_loss_frame",
        "persistent_critical_frame",
        "lost_frame",
        "risk_trigger_frame",
        "risk_classification",
        "risk_delay_frames",
        "lost_classification",
        "lost_delay_frames",
        "final_state",
    ]

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    print("PHASE 1F — L2 ALL-EVENT EVALUATION")
    print("=" * 55)

    print("\nPERSISTENT-CRITICAL / RECOVERY TRIGGER")
    print(
        f"TP={risk_counts['TP']} "
        f"TN={risk_counts['TN']} "
        f"FP={risk_counts['FP']} "
        f"FN={risk_counts['FN']}"
    )

    if risk_delays:
        print(
            "Mean trigger timing vs reference: "
            f"{sum(risk_delays)/len(risk_delays):.2f} frames"
        )

    print("\nSTRICT LOST STATE")
    print(
        f"TP={lost_counts['TP']} "
        f"TN={lost_counts['TN']} "
        f"FP={lost_counts['FP']} "
        f"FN={lost_counts['FN']}"
    )

    if lost_delays:
        print(
            "Mean LOST timing vs reference: "
            f"{sum(lost_delays)/len(lost_delays):.2f} frames"
        )

    print("\nIMPORTANT CASES")
    print("-" * 55)

    for row in rows:

        if (
            row["risk_classification"] in {"FP", "FN"}
            or row["lost_classification"] in {"FP", "FN"}
            or row["persistent_critical_frame"] is not None
        ):
            print(
                f"{row['event_id']} | "
                f"{row['tracker']:7} | "
                f"ref={row['reference_loss_frame']} | "
                f"critical={row['persistent_critical_frame']} | "
                f"lost={row['lost_frame']} | "
                f"risk={row['risk_classification']} | "
                f"strict={row['lost_classification']}"
            )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
