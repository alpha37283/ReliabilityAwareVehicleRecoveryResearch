import csv
from pathlib import Path

from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.annotation_candidate_generator import get_frame_candidates
from recovery.real_candidate_generator import filter_real_candidates


ROOT = Path(__file__).resolve().parents[2]

REFERENCE = ROOT / "results/phase1/phase1e_candidate_recall.csv"

OUTPUT = (
    ROOT
    / "results/phase1/phase1g_real_candidate_filtering.csv"
)

IMAGE_ROOT = ROOT / "data/raw/UA-DETRAC/images"

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

IMGSZ = 960
CONF = 0.15
MATCH_IOU = 0.30


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

    aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    bb = max(0.0, bx2-bx1) * max(0.0, by2-by1)

    union = aa + bb - inter

    return inter / union if union > 0 else 0.0


def best_iou(target_box, candidates):
    return max(
        (
            iou(
                target_box,
                candidate["box_xyxy"],
            )
            for candidate in candidates
        ),
        default=0.0,
    )


def main():
    model = YOLO("yolo11n.pt")

    with REFERENCE.open(newline="") as f:
        eligible = list(csv.DictReader(f))

    rows = []

    print("PHASE 1G — REAL CANDIDATE FILTERING")
    print("=" * 78)

    for ref in eligible:
        event_id = ref["event_id"]

        event = load_event(event_id)

        sequence = event["sequence"]
        target_id = int(event["target_id"])
        frame = int(event["reappearance_frame"])

        selection_box = event["selection_box"]

        if isinstance(selection_box, dict):
            reference_box = selection_box["xyxy"]
        else:
            reference_box = selection_box

        image_path = (
            IMAGE_ROOT
            / sequence
            / f"img{frame:05d}.jpg"
        )

        xml_path = XML_ROOT / f"{sequence}.xml"

        gt = get_frame_candidates(
            str(xml_path),
            frame,
        )

        target = next(
            g for g in gt
            if int(g["target_id"]) == target_id
        )

        target_box = target["box_xyxy"]

        result = model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=CONF,
            device=0,
            classes=[2, 3, 5, 7],
            verbose=False,
        )[0]

        raw_candidates = []

        for box in result.boxes:
            cls_id = int(box.cls[0])

            raw_candidates.append({
                "box_xyxy":
                    box.xyxy[0].tolist(),
                "confidence":
                    float(box.conf[0]),
                "class_id":
                    cls_id,
                "class_name":
                    result.names[cls_id],
            })

        filtered_candidates = filter_real_candidates(
            raw_candidates,
            reference_box,
        )

        raw_best = best_iou(
            target_box,
            raw_candidates,
        )

        filtered_best = best_iou(
            target_box,
            filtered_candidates,
        )

        raw_target = (
            raw_best >= MATCH_IOU
        )

        filtered_target = (
            filtered_best >= MATCH_IOU
        )

        reduction = (
            1.0
            - len(filtered_candidates)
            / len(raw_candidates)
            if raw_candidates
            else 0.0
        )

        rows.append({
            "event_id": event_id,
            "sequence": sequence,
            "target_id": target_id,
            "frame": frame,

            "raw_candidate_count":
                len(raw_candidates),

            "filtered_candidate_count":
                len(filtered_candidates),

            "candidate_reduction":
                round(reduction, 4),

            "raw_target_best_iou":
                round(raw_best, 4),

            "filtered_target_best_iou":
                round(filtered_best, 4),

            "raw_target_retained":
                raw_target,

            "filtered_target_retained":
                filtered_target,
        })

        print(
            f"{event_id} | "
            f"raw={len(raw_candidates):2d} | "
            f"filtered={len(filtered_candidates):2d} | "
            f"reduction={reduction*100:5.1f}% | "
            f"target_raw={raw_target} | "
            f"target_filtered={filtered_target}"
        )

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    raw_hits = sum(
        row["raw_target_retained"]
        for row in rows
    )

    filtered_hits = sum(
        row["filtered_target_retained"]
        for row in rows
    )

    raw_total = sum(
        row["raw_candidate_count"]
        for row in rows
    )

    filtered_total = sum(
        row["filtered_candidate_count"]
        for row in rows
    )

    total_reduction = (
        1.0 - filtered_total / raw_total
        if raw_total else 0.0
    )

    print()
    print("-" * 78)

    print(
        f"Raw target recall: "
        f"{raw_hits}/{len(rows)} = "
        f"{raw_hits/len(rows):.3f}"
    )

    print(
        f"Filtered target recall: "
        f"{filtered_hits}/{len(rows)} = "
        f"{filtered_hits/len(rows):.3f}"
    )

    print(
        f"Candidate count: "
        f"{raw_total} -> {filtered_total}"
    )

    print(
        f"Overall candidate reduction: "
        f"{total_reduction*100:.1f}%"
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
