import csv
import json
from pathlib import Path

from evaluation.event_loader import load_event


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results/phase1"
CANDIDATE_FILE = RESULTS_ROOT / "phase1e_candidate_recall.csv"
OUTPUT_FILE = RESULTS_ROOT / "phase1e_event_metrics.csv"

TRACKERS = ["sam21", "dam4sam"]
LOSS_STREAK = 3
RECOVERY_STREAK = 3


def as_bool(value):
    return str(value).strip().lower() == "true"


def first_streak(rows, start_index, desired_value, length):
    streak = 0
    streak_start = None

    for i in range(start_index, len(rows)):
        value = rows[i]["target_match"]

        if value == desired_value:
            if streak == 0:
                streak_start = i
            streak += 1

            if streak >= length:
                return streak_start
        else:
            streak = 0
            streak_start = None

    return None


def load_candidate_metrics():
    result = {}

    if not CANDIDATE_FILE.exists():
        return result

    with CANDIDATE_FILE.open(newline="") as f:
        for row in csv.DictReader(f):
            result[row["event_id"]] = row

    return result


def load_eval_rows(path):
    rows = []

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame_number": int(row["frame_number"]),
                "target_annotated": as_bool(row["target_annotated"]),
                "prediction_present": as_bool(row["prediction_present"]),
                "target_iou": float(row["target_iou"]),
                "target_match": as_bool(row["target_match"]),
                "best_other_id": row["best_other_id"],
                "best_other_iou": float(row["best_other_iou"]),
                "possible_switch": as_bool(row["possible_switch"]),
            })

    return rows


def evaluate_run(event_id, tracker, candidate_metrics):
    event = load_event(event_id)

    folder = RESULTS_ROOT / event_id / tracker
    eval_path = folder / "evaluation_frames.csv"
    metadata_path = folder / "run_metadata.json"

    rows = load_eval_rows(eval_path)
    metadata = json.loads(metadata_path.read_text())

    occlusion_start = event.get("occlusion_start")
    reappearance = event.get("reappearance_frame")

    if occlusion_start not in (None, "", "None"):
        occlusion_start = int(occlusion_start)
    else:
        occlusion_start = None

    if reappearance not in (None, "", "None"):
        reappearance = int(reappearance)
    else:
        reappearance = None

    # --------------------------------------------------
    # Pre-occlusion tracking
    # --------------------------------------------------
    pre_rows = []

    if occlusion_start is not None:
        pre_rows = [
            r for r in rows
            if r["frame_number"] < occlusion_start
        ]

    if pre_rows:
        pre_matches = sum(r["target_match"] for r in pre_rows)
        pre_tracking_rate = pre_matches / len(pre_rows)
        pre_occlusion_tracking = pre_tracking_rate >= 0.8
    else:
        pre_tracking_rate = None
        pre_occlusion_tracking = None

        # --------------------------------------------------
    # Loss detection
    # Search only inside the occlusion/recovery interval.
    # If a verified reappearance exists, loss must occur
    # before that frame.
    # --------------------------------------------------

    loss_search_indices = []

    for i, r in enumerate(rows):
        frame = r["frame_number"]

        if occlusion_start is not None and frame < occlusion_start:
            continue

        if reappearance is not None and frame >= reappearance:
            continue

        loss_search_indices.append(i)

    loss_index = None

    streak = 0
    streak_start = None

    for i in loss_search_indices:
        if rows[i]["target_match"] is False:
            if streak == 0:
                streak_start = i

            streak += 1

            if streak >= LOSS_STREAK:
                loss_index = streak_start
                break
        else:
            streak = 0
            streak_start = None

    if loss_index is None:
        loss_frame = None
        continuous_tracking = True
    else:
        loss_frame = rows[loss_index]["frame_number"]
        continuous_tracking = False

        # --------------------------------------------------
    # Recovery detection
    # Recovery is only meaningful after a detected loss
    # and cannot begin before verified reappearance.
    # --------------------------------------------------

    recovery_index = None
    recovery_frame = None

    if loss_index is not None and reappearance is not None:

        recovery_start_index = None

        for i, r in enumerate(rows):
            if r["frame_number"] >= reappearance:
                recovery_start_index = i
                break

        if recovery_start_index is not None:
            recovery_index = first_streak(
                rows,
                start_index=recovery_start_index,
                desired_value=True,
                length=RECOVERY_STREAK,
            )

            if recovery_index is not None:
                recovery_frame = rows[recovery_index]["frame_number"]

    # --------------------------------------------------
    # Recovery outcome
    # --------------------------------------------------
    if continuous_tracking:
        recovery_status = "not_applicable_continuous_tracking"
        recovery_delay = None

    elif recovery_frame is not None:
        recovery_status = "automatic_correct_recovery"

        if reappearance is not None:
            recovery_delay = recovery_frame - reappearance
        else:
            recovery_delay = None

    else:
        if reappearance is not None:
            recovery_status = "automatic_no_recovery"
        else:
            recovery_status = "loss_no_verified_reappearance"

        recovery_delay = None

    # --------------------------------------------------
    # Switch warnings
    # These are NOT confirmed ID switches.
    # --------------------------------------------------
    switch_frames = [
        r["frame_number"]
        for r in rows
        if r["possible_switch"]
    ]

    switch_warning_count = len(switch_frames)

    if switch_frames:
        first_switch_warning = switch_frames[0]
        id_switch_status = "manual_review_required"
    else:
        first_switch_warning = None
        id_switch_status = "no_automatic_warning"

    # --------------------------------------------------
    # Candidate reference metrics
    # --------------------------------------------------
    candidate = candidate_metrics.get(event_id)

    if candidate is None:
        candidate_retained = None
        candidate_count = None
        distractor_count = None
    else:
        candidate_retained = as_bool(
            candidate["true_target_in_plausible"]
        )
        candidate_count = int(
            candidate["plausible_candidate_count"]
        )
        distractor_count = int(
            candidate["distractor_count"]
        )

    # --------------------------------------------------
    # Runtime
    # --------------------------------------------------
    latency_ms = float(
        metadata["average_tracker_runtime_ms"]
    )

    fps = 1000.0 / latency_ms if latency_ms > 0 else None

    return {
        "event_id": event_id,
        "tracker": tracker,

        "pre_occlusion_tracking": pre_occlusion_tracking,
        "pre_occlusion_tracking_rate": pre_tracking_rate,

        "continuous_tracking": continuous_tracking,

        "loss_frame": loss_frame,

        "switch_warning_count": switch_warning_count,
        "first_switch_warning_frame": first_switch_warning,
        "id_switch_status": id_switch_status,

        "recorded_reappearance_frame": reappearance,

        "recovery_status": recovery_status,
        "recovery_frame": recovery_frame,
        "recovery_delay_frames": recovery_delay,

        "candidate_retained": candidate_retained,
        "candidate_count": candidate_count,
        "distractor_count": distractor_count,

        "latency_ms": latency_ms,
        "fps": fps,
    }


def main():
    candidate_metrics = load_candidate_metrics()

    results = []

    for i in range(1, 13):
        event_id = f"E{i:03d}"

        for tracker in TRACKERS:
            result = evaluate_run(
                event_id,
                tracker,
                candidate_metrics,
            )

            results.append(result)

            print(
                f"{event_id} | {tracker:7} | "
                f"continuous={result['continuous_tracking']} | "
                f"loss={result['loss_frame']} | "
                f"recovery={result['recovery_frame']} | "
                f"status={result['recovery_status']} | "
                f"switch_warnings={result['switch_warning_count']} | "
                f"candidate_retained={result['candidate_retained']}"
            )

    fields = [
        "event_id",
        "tracker",
        "pre_occlusion_tracking",
        "pre_occlusion_tracking_rate",
        "continuous_tracking",
        "loss_frame",
        "switch_warning_count",
        "first_switch_warning_frame",
        "id_switch_status",
        "recorded_reappearance_frame",
        "recovery_status",
        "recovery_frame",
        "recovery_delay_frames",
        "candidate_retained",
        "candidate_count",
        "distractor_count",
        "latency_ms",
        "fps",
    ]

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Tracker-event runs evaluated: {len(results)}")


if __name__ == "__main__":
    main()
