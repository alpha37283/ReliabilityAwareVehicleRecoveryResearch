import csv
from pathlib import Path

from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.real_candidate_generator import filter_real_candidates


ROOT = Path(__file__).resolve().parents[2]

ELIGIBLE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

IMAGE_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/images"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h2_candidate_duplicate_diagnostic.csv"
)

THRESHOLDS = [0.50, 0.70, 0.90]


def get_xyxy(box):
    if isinstance(box, dict):
        return box["xyxy"]
    return box


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

    return inter / union if union > 0 else 0.0


def main():

    detector = YOLO("yolo11n.pt")

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    rows = []

    print(
        "PHASE 1H-2 — CANDIDATE DUPLICATE / OVERLAP DIAGNOSTIC"
    )
    print("=" * 110)

    for item in events:

        event_id = item["event_id"]
        event = load_event(event_id)

        sequence = event["sequence"]

        reappearance = int(
            event["reappearance_frame"]
        )

        selection_box = get_xyxy(
            event["selection_box"]
        )

        image_path = (
            IMAGE_ROOT
            / sequence
            / f"img{reappearance:05d}.jpg"
        )

        # --------------------------------------------------
        # Frozen Phase 1G detector
        # --------------------------------------------------

        result = detector.predict(
            source=str(image_path),
            imgsz=640,
            conf=0.20,
            device=0,
            classes=[2, 3, 5, 7],
            verbose=False,
        )[0]

        raw = []

        for box in result.boxes:

            class_id = int(box.cls[0])

            raw.append({
                "box_xyxy":
                    box.xyxy[0].tolist(),

                "confidence":
                    float(box.conf[0]),

                "class_id":
                    class_id,

                "class_name":
                    result.names[class_id],
            })

        candidates = filter_real_candidates(
            raw,
            selection_box,
        )

        # --------------------------------------------------
        # Pairwise overlap analysis
        # --------------------------------------------------

        pair_ious = []

        for i in range(len(candidates)):

            for j in range(i + 1, len(candidates)):

                overlap = box_iou(
                    candidates[i]["box_xyxy"],
                    candidates[j]["box_xyxy"],
                )

                pair_ious.append({
                    "i": i,
                    "j": j,
                    "iou": overlap,
                })

        counts = {}

        for threshold in THRESHOLDS:

            counts[threshold] = sum(
                pair["iou"] >= threshold
                for pair in pair_ious
            )

        max_pair = (
            max(
                pair_ious,
                key=lambda x: x["iou"],
            )
            if pair_ious
            else None
        )

        max_iou = (
            max_pair["iou"]
            if max_pair
            else 0.0
        )

        # Count how many candidate boxes participate in
        # at least one strong-overlap pair (IoU >= 0.70).
        participating = set()

        for pair in pair_ious:

            if pair["iou"] >= 0.70:

                participating.add(
                    pair["i"]
                )

                participating.add(
                    pair["j"]
                )

        rows.append({
            "event_id":
                event_id,

            "candidate_count":
                len(candidates),

            "pair_count":
                len(pair_ious),

            "pairs_iou_ge_050":
                counts[0.50],

            "pairs_iou_ge_070":
                counts[0.70],

            "pairs_iou_ge_090":
                counts[0.90],

            "candidates_in_iou_ge_070_pairs":
                len(participating),

            "maximum_pair_iou":
                round(max_iou, 6),
        })

        print(
            f"{event_id} | "
            f"candidates={len(candidates):2d} | "
            f">=.50={counts[0.50]:2d} | "
            f">=.70={counts[0.70]:2d} | "
            f">=.90={counts[0.90]:2d} | "
            f"boxes_in_.70={len(participating):2d} | "
            f"maxIoU={max_iou:.3f}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

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

    print()
    print("-" * 110)

    for threshold, column in [
        (0.50, "pairs_iou_ge_050"),
        (0.70, "pairs_iou_ge_070"),
        (0.90, "pairs_iou_ge_090"),
    ]:

        events_with_duplicates = sum(
            int(row[column]) > 0
            for row in rows
        )

        total_pairs = sum(
            int(row[column])
            for row in rows
        )

        print(
            f"IoU >= {threshold:.2f}: "
            f"{events_with_duplicates}/{len(rows)} events, "
            f"{total_pairs} overlapping candidate pairs"
        )

    print()
    print(
        "No duplicate-removal threshold is frozen "
        "from this diagnostic."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
