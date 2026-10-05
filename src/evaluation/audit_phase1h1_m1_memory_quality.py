import csv
from pathlib import Path
from statistics import mean

from evaluation.event_loader import load_event
from recovery.recent_memory import load_recent_memory
from recovery.annotation_candidate_generator import get_frame_candidates


ROOT = Path(__file__).resolve().parents[2]

ELIGIBLE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

IMAGE_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/images"
)

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h1_m1_memory_quality.csv"
)

MEMORY_CAPACITY = 5


def get_xyxy(box):
    if isinstance(box, dict):
        return box["xyxy"]
    return box


def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def quality_label(value):

    if value >= 0.50:
        return "strong"

    if value >= 0.30:
        return "weak"

    return "contaminated"


def main():

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    rows = []

    print("PHASE 1H-1 — M1 MEMORY QUALITY AUDIT")
    print("=" * 100)

    for row in events:

        event_id = row["event_id"]

        event = load_event(event_id)

        sequence = event["sequence"]
        target_id = int(event["target_id"])

        selection_frame = int(
            event["selection_frame"]
        )

        cutoff = int(
            event["occlusion_start"]
        )

        predictions = (
            ROOT
            / "results/phase1"
            / event_id
            / "sam21"
            / "predictions.csv"
        )

        memory = load_recent_memory(
            predictions_csv=predictions,
            image_dir=IMAGE_ROOT / sequence,
            selection_frame=selection_frame,
            memory_cutoff_frame=cutoff,
            selection_box=get_xyxy(
                event["selection_box"]
            ),
            capacity=MEMORY_CAPACITY,
        )

        xml_path = (
            XML_ROOT
            / f"{sequence}.xml"
        )

        event_ious = []

        print()
        print(f"{event_id}")
        print("-" * 100)

        for entry in memory:

            frame = int(
                entry["frame_number"]
            )

            gt_objects = get_frame_candidates(
                str(xml_path),
                frame,
            )

            target_gt = next(
                (
                    obj
                    for obj in gt_objects
                    if int(obj["target_id"])
                    == target_id
                ),
                None,
            )

            if target_gt is None:

                rows.append({
                    "event_id": event_id,
                    "frame_number": frame,
                    "source": entry["source"],
                    "tracker_gt_iou": "",
                    "quality": "gt_missing",
                })

                print(
                    f"frame={frame} | "
                    f"source={entry['source']:14} | "
                    f"GT missing"
                )

                continue

            score = iou(
                entry["box_xyxy"],
                target_gt["box_xyxy"],
            )

            label = quality_label(score)

            event_ious.append(score)

            rows.append({
                "event_id": event_id,
                "frame_number": frame,
                "source": entry["source"],
                "tracker_gt_iou":
                    round(score, 4),
                "quality": label,
            })

            print(
                f"frame={frame} | "
                f"source={entry['source']:14} | "
                f"IoU={score:.4f} | "
                f"{label}"
            )

        tracker_rows = [
            r for r in rows
            if (
                r["event_id"] == event_id
                and r["source"]
                == "recent_tracker"
                and str(
                    r["tracker_gt_iou"]
                ).strip()
            )
        ]

        if tracker_rows:

            vals = [
                float(r["tracker_gt_iou"])
                for r in tracker_rows
            ]

            contaminated = sum(
                v < 0.30
                for v in vals
            )

            print(
                f"Recent tracker memory: "
                f"mean IoU={mean(vals):.4f} | "
                f"contaminated="
                f"{contaminated}/{len(vals)}"
            )

        else:

            print(
                "Recent tracker memory: none"
            )

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "frame_number",
                "source",
                "tracker_gt_iou",
                "quality",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    recent_rows = [
        r for r in rows
        if (
            r["source"] == "recent_tracker"
            and str(
                r["tracker_gt_iou"]
            ).strip()
        )
    ]

    values = [
        float(r["tracker_gt_iou"])
        for r in recent_rows
    ]

    strong = sum(
        v >= 0.50
        for v in values
    )

    weak = sum(
        0.30 <= v < 0.50
        for v in values
    )

    contaminated = sum(
        v < 0.30
        for v in values
    )

    print()
    print("=" * 100)
    print("OVERALL RECENT-TRACKER MEMORY QUALITY")
    print("-" * 100)

    print(
        f"Recent tracker entries: "
        f"{len(values)}"
    )

    if values:

        print(
            f"Mean GT IoU: "
            f"{mean(values):.4f}"
        )

        print(
            f"Strong >=0.50: "
            f"{strong}/{len(values)}"
        )

        print(
            f"Weak 0.30-0.50: "
            f"{weak}/{len(values)}"
        )

        print(
            f"Contaminated <0.30: "
            f"{contaminated}/{len(values)}"
        )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
