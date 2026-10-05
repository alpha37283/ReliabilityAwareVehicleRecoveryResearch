import csv
import math
from pathlib import Path
from statistics import mean

from evaluation.event_loader import load_event


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/phase1"
REFERENCE = RESULTS / "phase1e_final_metrics.csv"


def as_bool(x):
    return str(x).strip().lower() == "true"


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = aa + ba - inter

    return inter / union if union > 0 else 0.0


def center_jump(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2

    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    aw = ax2 - ax1
    ah = ay2 - ay1

    diag = math.sqrt(aw * aw + ah * ah)

    if diag <= 0:
        return 0.0

    jump = math.sqrt(
        (bcx - acx) ** 2 +
        (bcy - acy) ** 2
    )

    return jump / diag


def percentile(values, p):
    if not values:
        return None

    values = sorted(values)

    k = (len(values) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))

    if lo == hi:
        return values[lo]

    return (
        values[lo] * (hi - k)
        + values[hi] * (k - lo)
    )


def load_reference():
    out = {}

    with REFERENCE.open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["event_id"], row["tracker"])] = row

    return out


def analyze_run(event_id, tracker, ref_row):
    event = load_event(event_id)

    occlusion_start = event.get("occlusion_start")
    reappearance = event.get("reappearance_frame")

    path = RESULTS / event_id / tracker / "predictions.csv"

    ious = []
    jumps = []

    missing = 0
    total = 0

    previous_present_box = None

    with path.open(newline="") as f:
        for row in csv.DictReader(f):

            frame = int(row["frame_number"])

            if occlusion_start is not None and frame < occlusion_start:
                continue

            if reappearance is not None and frame >= reappearance:
                break

            total += 1

            present = as_bool(row["present"])

            if not present:
                missing += 1
                continue

            box = [
                float(row["box_x1"]),
                float(row["box_y1"]),
                float(row["box_x2"]),
                float(row["box_y2"]),
            ]

            if previous_present_box is not None:
                ious.append(
                    box_iou(previous_present_box, box)
                )

                jumps.append(
                    center_jump(previous_present_box, box)
                )

            previous_present_box = box

    has_reference_loss = bool(
        ref_row["loss_frame"].strip()
    )

    return {
        "has_loss": has_reference_loss,
        "ious": ious,
        "jumps": jumps,
        "missing": missing,
        "total": total,
    }


def summarize(name, runs):
    ious = []
    jumps = []

    missing = 0
    total = 0

    for r in runs:
        ious.extend(r["ious"])
        jumps.extend(r["jumps"])
        missing += r["missing"]
        total += r["total"]

    print()
    print(name)
    print("-" * 50)
    print(f"Runs: {len(runs)}")
    print(f"Frames examined: {total}")
    print(f"Missing predictions: {missing}")

    if ious:
        print(
            "Temporal IoU: "
            f"p01={percentile(ious,0.01):.3f} | "
            f"p05={percentile(ious,0.05):.3f} | "
            f"p10={percentile(ious,0.10):.3f} | "
            f"median={percentile(ious,0.50):.3f}"
        )

    if jumps:
        print(
            "Center jump: "
            f"median={percentile(jumps,0.50):.3f} | "
            f"p90={percentile(jumps,0.90):.3f} | "
            f"p95={percentile(jumps,0.95):.3f} | "
            f"p99={percentile(jumps,0.99):.3f}"
        )


def main():
    reference = load_reference()

    stable_runs = []
    loss_runs = []

    for i in range(1, 13):
        event_id = f"E{i:03d}"

        for tracker in ["sam21", "dam4sam"]:

            ref = reference[(event_id, tracker)]

            result = analyze_run(
                event_id,
                tracker,
                ref,
            )

            if result["has_loss"]:
                loss_runs.append(result)
            else:
                stable_runs.append(result)

    print("L2 SIGNAL DISTRIBUTION ANALYSIS")
    print("=" * 50)

    summarize(
        "REFERENCE NO-LOSS RUNS",
        stable_runs,
    )

    summarize(
        "REFERENCE LOSS RUNS",
        loss_runs,
    )


if __name__ == "__main__":
    main()
