import csv
from pathlib import Path

from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.annotation_candidate_generator import get_frame_candidates
from recovery.real_candidate_generator import filter_real_candidates


ROOT = Path(__file__).resolve().parents[2]

REFERENCE = ROOT / "results/phase1/phase1e_candidate_recall.csv"

IMAGE_ROOT = ROOT / "data/raw/UA-DETRAC/images"

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

MATCH_IOU = 0.30

SETTINGS = [
    (640, 0.15),
    (640, 0.20),
    (960, 0.15),
    (960, 0.20),
    (960, 0.25),
]


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


def main():
    model = YOLO("yolo11n.pt")

    with REFERENCE.open(newline="") as f:
        eligible = list(csv.DictReader(f))

    print("PHASE 1G — DETECTOR SETTING SWEEP")
    print("=" * 78)
    print(
        "imgsz conf target_recall raw_candidates "
        "filtered_candidates reduction"
    )

    for imgsz, conf in SETTINGS:

        target_hits = 0
        raw_total = 0
        filtered_total = 0

        for ref in eligible:

            event = load_event(ref["event_id"])

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
                imgsz=imgsz,
                conf=conf,
                device=0,
                classes=[2, 3, 5, 7],
                verbose=False,
            )[0]

            raw = []

            for box in result.boxes:
                cls_id = int(box.cls[0])

                raw.append({
                    "box_xyxy": box.xyxy[0].tolist(),
                    "confidence": float(box.conf[0]),
                    "class_id": cls_id,
                    "class_name": result.names[cls_id],
                })

            filtered = filter_real_candidates(
                raw,
                reference_box,
            )

            best = max(
                (
                    iou(
                        target_box,
                        c["box_xyxy"],
                    )
                    for c in filtered
                ),
                default=0.0,
            )

            if best >= MATCH_IOU:
                target_hits += 1

            raw_total += len(raw)
            filtered_total += len(filtered)

        reduction = (
            1.0 - filtered_total / raw_total
            if raw_total else 0.0
        )

        print(
            f"{imgsz:<5} "
            f"{conf:<4.2f} "
            f"{target_hits}/{len(eligible):<8} "
            f"{raw_total:<14} "
            f"{filtered_total:<19} "
            f"{reduction*100:5.1f}%"
        )


if __name__ == "__main__":
    main()
