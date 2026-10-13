import csv
from pathlib import Path

from evaluation.event_loader import load_event
from recovery.annotation_candidate_generator import get_frame_candidates


ROOT = Path(__file__).resolve().parents[2]

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1i_f1_preocclusion_diagnostic.csv"
)

EVENT_IDS = [
    f"E{i:03d}"
    for i in range(1, 13)
]

TRACKERS = [
    "sam21",
    "dam4sam",
]

# Diagnostic threshold only.
LOW_IOU = 0.30

# Three consecutive poor-overlap frames make a case
# worth manual F1 inspection. This does NOT assign F1.
SUSPECT_RUN = 3


def parse_bool(value):

    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


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

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - inter

    return (
        inter / union
        if union > 0
        else 0.0
    )


def load_predictions(event_id, tracker):

    path = (
        ROOT
        / "results/phase1"
        / event_id
        / tracker
        / "predictions.csv"
    )

    lookup = {}

    if not path.exists():
        return lookup

    with path.open(newline="") as f:

        for row in csv.DictReader(f):

            frame = int(
                row["frame_number"]
            )

            if not parse_bool(
                row["present"]
            ):
                lookup[frame] = None
                continue

            try:
                box = [
                    float(row["box_x1"]),
                    float(row["box_y1"]),
                    float(row["box_x2"]),
                    float(row["box_y2"]),
                ]
            except Exception:
                lookup[frame] = None
                continue

            if (
                box[2] <= box[0]
                or box[3] <= box[1]
            ):
                lookup[frame] = None
                continue

            lookup[frame] = box

    return lookup


def longest_low_run(values):

    longest = 0
    current = 0
    start = None
    best_start = None

    for frame, low in values:

        if low:

            if current == 0:
                start = frame

            current += 1

            if current > longest:
                longest = current
                best_start = start

        else:
            current = 0
            start = None

    return longest, best_start


def main():

    rows = []

    print(
        "PHASE 1I — F1 PRE-OCCLUSION DRIFT DIAGNOSTIC"
    )
    print("=" * 120)

    for event_id in EVENT_IDS:

        event = load_event(
            event_id
        )

        sequence = event["sequence"]

        target_id = int(
            event["target_id"]
        )

        selection = int(
            event["selection_frame"]
        )

        occlusion_start = int(
            event["occlusion_start"]
        )

        xml_path = (
            XML_ROOT
            / f"{sequence}.xml"
        )

        for tracker in TRACKERS:

            predictions = load_predictions(
                event_id,
                tracker,
            )

            ious = []
            low_flags = []

            missing_prediction = 0
            missing_gt = 0

            for frame in range(
                selection,
                occlusion_start,
            ):

                objects = get_frame_candidates(
                    str(xml_path),
                    frame,
                )

                target_objects = [
                    obj
                    for obj in objects
                    if int(
                        obj["target_id"]
                    ) == target_id
                ]

                if not target_objects:
                    missing_gt += 1
                    continue

                gt_box = target_objects[0][
                    "box_xyxy"
                ]

                pred_box = predictions.get(
                    frame
                )

                if pred_box is None:

                    missing_prediction += 1

                    # Missing prediction is tracked separately.
                    # Do not pretend it has geometric IoU.
                    low_flags.append(
                        (frame, True)
                    )

                    continue

                overlap = box_iou(
                    pred_box,
                    gt_box,
                )

                ious.append(
                    (frame, overlap)
                )

                low_flags.append(
                    (
                        frame,
                        overlap < LOW_IOU,
                    )
                )

            low_iou_frames = [
                frame
                for frame, overlap in ious
                if overlap < LOW_IOU
            ]

            longest_run, run_start = (
                longest_low_run(
                    low_flags
                )
            )

            valid_ious = [
                overlap
                for _, overlap in ious
            ]

            mean_iou = (
                sum(valid_ious)
                / len(valid_ious)
                if valid_ious
                else None
            )

            min_iou = (
                min(valid_ious)
                if valid_ious
                else None
            )

            suspect = (
                longest_run
                >= SUSPECT_RUN
            )

            row = {
                "event_id":
                    event_id,

                "tracker":
                    tracker,

                "selection_frame":
                    selection,

                "occlusion_start":
                    occlusion_start,

                "evaluated_preocclusion_frames":
                    (
                        occlusion_start
                        - selection
                    ),

                "valid_iou_frames":
                    len(valid_ious),

                "missing_prediction_frames":
                    missing_prediction,

                "missing_gt_frames":
                    missing_gt,

                "low_iou_frames":
                    len(low_iou_frames),

                "mean_iou":
                    (
                        round(mean_iou, 6)
                        if mean_iou is not None
                        else ""
                    ),

                "minimum_iou":
                    (
                        round(min_iou, 6)
                        if min_iou is not None
                        else ""
                    ),

                "longest_low_or_missing_run":
                    longest_run,

                "first_long_run_start":
                    (
                        run_start
                        if suspect
                        else ""
                    ),

                "f1_manual_review_suspect":
                    suspect,
            }

            rows.append(row)

            print(
                f"{event_id} / {tracker:8} | "
                f"frames={occlusion_start-selection:3d} | "
                f"meanIoU="
                f"{mean_iou if mean_iou is not None else 0:.3f} | "
                f"minIoU="
                f"{min_iou if min_iou is not None else 0:.3f} | "
                f"low={len(low_iou_frames):3d} | "
                f"missing={missing_prediction:3d} | "
                f"maxrun={longest_run:3d} | "
                f"suspect={suspect}"
            )

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    suspects = [
        row
        for row in rows
        if row[
            "f1_manual_review_suspect"
        ]
    ]

    print()
    print("-" * 120)

    print(
        f"F1 manual-review suspects: "
        f"{len(suspects)}/24"
    )

    for row in suspects:

        print(
            f"  {row['event_id']} / "
            f"{row['tracker']} | "
            f"run={row['longest_low_or_missing_run']} | "
            f"start={row['first_long_run_start']}"
        )

    print()
    print(
        "IMPORTANT: this script does NOT assign F1."
    )

    print(
        "A suspect only means sustained poor/missing "
        "tracker overlap occurred BEFORE occlusion_start."
    )

    print(
        "Any suspect must be visually reviewed on "
        "the original frames before taxonomy labeling."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
