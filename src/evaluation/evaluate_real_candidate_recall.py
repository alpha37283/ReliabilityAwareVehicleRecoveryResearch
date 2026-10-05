import csv
from pathlib import Path

from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.annotation_candidate_generator import get_frame_candidates


ROOT = Path(__file__).resolve().parents[2]

REFERENCE = ROOT / "results/phase1/phase1e_candidate_recall.csv"
OUTPUT = ROOT / "results/phase1/phase1g_real_candidate_recall.csv"

IMAGE_ROOT = ROOT / "data/raw/UA-DETRAC/images"
XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

IMGSZ = 960
CONF = 0.15
IOU_THRESHOLD = 0.30


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


def greedy_match(gt, detections, threshold):
    pairs = []

    for gi, g in enumerate(gt):
        for di, d in enumerate(detections):
            pairs.append(
                (
                    iou(g["box_xyxy"], d),
                    gi,
                    di,
                )
            )

    pairs.sort(reverse=True)

    used_gt = set()
    used_det = set()

    for score, gi, di in pairs:
        if score < threshold:
            break

        if gi in used_gt or di in used_det:
            continue

        used_gt.add(gi)
        used_det.add(di)

    return len(used_gt)


def main():
    model = YOLO("yolo11n.pt")

    with REFERENCE.open(newline="") as f:
        eligible = list(csv.DictReader(f))

    rows = []

    print("PHASE 1G — REAL CANDIDATE RECALL")
    print("=" * 75)

    for ref in eligible:
        event_id = ref["event_id"]

        event = load_event(event_id)

        sequence = event["sequence"]
        target_id = int(event["target_id"])
        frame = int(event["reappearance_frame"])

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

        result = model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=CONF,
            device=0,
            classes=[2, 3, 5, 7],
            verbose=False,
        )[0]

        detections = [
            box.xyxy[0].tolist()
            for box in result.boxes
        ]

        matched = greedy_match(
            gt,
            detections,
            IOU_THRESHOLD,
        )

        frame_recall = (
            matched / len(gt)
            if gt else 0.0
        )

        target = next(
            (
                g for g in gt
                if int(g["target_id"]) == target_id
            ),
            None,
        )

        if target is None:
            best_target_iou = None
            target_found = False
        else:
            best_target_iou = max(
                (
                    iou(
                        target["box_xyxy"],
                        det,
                    )
                    for det in detections
                ),
                default=0.0,
            )

            target_found = (
                best_target_iou >= IOU_THRESHOLD
            )

        inference_ms = (
            result.speed.get("inference")
            if result.speed
            else None
        )

        rows.append({
            "event_id": event_id,
            "sequence": sequence,
            "target_id": target_id,
            "frame": frame,
            "gt_vehicle_count": len(gt),
            "detector_count": len(detections),
            "matched_gt_count": matched,
            "frame_recall_at_030": round(frame_recall, 4),
            "target_best_iou": (
                round(best_target_iou, 4)
                if best_target_iou is not None
                else ""
            ),
            "target_retained": target_found,
            "inference_ms": (
                round(inference_ms, 3)
                if inference_ms is not None
                else ""
            ),
        })

        print(
            f"{event_id} | "
            f"frame={frame} | "
            f"GT={len(gt):2d} | "
            f"det={len(detections):2d} | "
            f"recall={matched}/{len(gt)} "
            f"({frame_recall:.2f}) | "
            f"target={target_found} | "
            f"IoU={best_target_iou:.3f}"
        )

    fields = list(rows[0].keys())

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    target_hits = sum(
        row["target_retained"]
        for row in rows
    )

    mean_frame_recall = sum(
        row["frame_recall_at_030"]
        for row in rows
    ) / len(rows)

    print()
    print("-" * 75)
    print(
        f"Target candidate recall: "
        f"{target_hits}/{len(rows)} = "
        f"{target_hits/len(rows):.3f}"
    )
    print(
        f"Mean frame vehicle recall: "
        f"{mean_frame_recall:.3f}"
    )
    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
