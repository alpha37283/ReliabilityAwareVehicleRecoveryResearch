import csv
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder
from recovery.recent_memory import load_recent_memory
from recovery.annotation_candidate_generator import get_frame_candidates
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

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h3_confidence_consistency.csv"
)

TARGET_IOU = 0.30


def get_xyxy(box):
    return box["xyxy"] if isinstance(box, dict) else box


def clamp_box(box, width, height):

    x1, y1, x2, y2 = box

    return [
        int(max(0, min(width, round(float(x1))))),
        int(max(0, min(height, round(float(y1))))),
        int(max(0, min(width, round(float(x2))))),
        int(max(0, min(height, round(float(y2))))),
    ]


def area(box):

    x1, y1, x2, y2 = box

    return (
        max(0.0, x2-x1)
        * max(0.0, y2-y1)
    )


def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = (
        max(0.0, ix2-ix1)
        * max(0.0, iy2-iy1)
    )

    union = (
        area(a)
        + area(b)
        - inter
    )

    return inter / union if union > 0 else 0.0


def main():

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    rows = []

    print(
        "PHASE 1H-3 — APPEARANCE / DETECTOR-CONFIDENCE CONSISTENCY"
    )
    print("=" * 125)

    for item in events:

        event_id = item["event_id"]
        event = load_event(event_id)

        sequence = event["sequence"]
        target_id = int(event["target_id"])

        selection_frame = int(
            event["selection_frame"]
        )

        occlusion_start = int(
            event["occlusion_start"]
        )

        reappearance = int(
            event["reappearance_frame"]
        )

        selection_box = get_xyxy(
            event["selection_box"]
        )

        predictions_csv = (
            ROOT
            / "results/phase1"
            / event_id
            / "sam21"
            / "predictions.csv"
        )

        # --------------------------------------------------
        # Frozen M1 positive memory
        # --------------------------------------------------

        memory = load_recent_memory(
            predictions_csv=predictions_csv,
            image_dir=IMAGE_ROOT / sequence,
            selection_frame=selection_frame,
            memory_cutoff_frame=occlusion_start,
            selection_box=selection_box,
            capacity=5,
        )

        memory_embeddings = torch.stack([
            encoder.encode(x["crop"])
            for x in memory
        ])

        # --------------------------------------------------
        # Frozen Phase 1G candidates
        # --------------------------------------------------

        image_path = (
            IMAGE_ROOT
            / sequence
            / f"img{reappearance:05d}.jpg"
        )

        image = Image.open(
            image_path
        ).convert("RGB")

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

        scored = []

        for candidate in candidates:

            box = clamp_box(
                candidate["box_xyxy"],
                image.width,
                image.height,
            )

            x1, y1, x2, y2 = box

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop(box)

            embedding = encoder.encode(crop)

            appearance_score = float(
                torch.max(
                    torch.mv(
                        memory_embeddings,
                        embedding,
                    )
                ).item()
            )

            scored.append({
                "box_xyxy": box,

                "appearance_score":
                    appearance_score,

                "detector_confidence":
                    float(candidate["confidence"]),
            })

        # --------------------------------------------------
        # Appearance ranking
        # --------------------------------------------------

        scored.sort(
            key=lambda x:
                x["appearance_score"],
            reverse=True,
        )

        for rank, candidate in enumerate(
            scored,
            start=1,
        ):
            candidate["appearance_rank"] = rank

        top1 = scored[0]

        # --------------------------------------------------
        # Detector-confidence ranking
        # --------------------------------------------------

        confidence_sorted = sorted(
            scored,
            key=lambda x:
                x["detector_confidence"],
            reverse=True,
        )

        for rank, candidate in enumerate(
            confidence_sorted,
            start=1,
        ):
            candidate["confidence_rank"] = rank

        max_confidence = (
            confidence_sorted[0][
                "detector_confidence"
            ]
        )

        top1_confidence = (
            top1["detector_confidence"]
        )

        confidence_gap = (
            max_confidence
            - top1_confidence
        )

        confidence_rank_fraction = (
            top1["confidence_rank"]
            / len(scored)
        )

        # 1.0 = highest-confidence detection
        # 0.0 = lowest-confidence detection
        if len(scored) > 1:

            confidence_percentile = (
                len(scored)
                - top1["confidence_rank"]
            ) / (
                len(scored) - 1
            )

        else:
            confidence_percentile = 1.0

        # --------------------------------------------------
        # GT only AFTER all runtime scoring/ranking
        # --------------------------------------------------

        xml_path = (
            XML_ROOT
            / f"{sequence}.xml"
        )

        gt = get_frame_candidates(
            str(xml_path),
            reappearance,
        )

        target_gt = next(
            obj
            for obj in gt
            if int(obj["target_id"]) == target_id
        )

        target_iou = iou(
            top1["box_xyxy"],
            target_gt["box_xyxy"],
        )

        correct = (
            target_iou >= TARGET_IOU
        )

        row = {
            "event_id":
                event_id,

            "top1_correct":
                correct,

            "candidate_count":
                len(scored),

            "top1_appearance_score":
                round(
                    top1["appearance_score"],
                    6,
                ),

            "top1_detector_confidence":
                round(
                    top1_confidence,
                    6,
                ),

            "top1_detector_confidence_rank":
                top1["confidence_rank"],

            "top1_confidence_rank_fraction":
                round(
                    confidence_rank_fraction,
                    6,
                ),

            "top1_confidence_percentile":
                round(
                    confidence_percentile,
                    6,
                ),

            "maximum_detector_confidence":
                round(
                    max_confidence,
                    6,
                ),

            "confidence_gap_to_frame_max":
                round(
                    confidence_gap,
                    6,
                ),

            "top1_target_iou":
                round(
                    target_iou,
                    4,
                ),
        }

        rows.append(row)

        print(
            f"{event_id} | "
            f"correct={str(correct):5} | "
            f"app={top1['appearance_score']:.4f} | "
            f"conf={top1_confidence:.3f} | "
            f"conf_rank="
            f"{top1['confidence_rank']:2d}/{len(scored):2d} | "
            f"percentile={confidence_percentile:.3f} | "
            f"frame_max={max_confidence:.3f} | "
            f"gap={confidence_gap:.3f}"
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

    # --------------------------------------------------
    # Correct vs wrong summary
    # --------------------------------------------------

    correct_rows = [
        r for r in rows
        if r["top1_correct"]
    ]

    wrong_rows = [
        r for r in rows
        if not r["top1_correct"]
    ]

    print()
    print("-" * 125)

    metrics = [
        (
            "top1_detector_confidence",
            "Absolute detector confidence",
        ),
        (
            "top1_confidence_percentile",
            "Within-frame confidence percentile",
        ),
        (
            "confidence_gap_to_frame_max",
            "Gap to frame maximum confidence",
        ),
    ]

    for key, label in metrics:

        c = [
            float(r[key])
            for r in correct_rows
        ]

        w = [
            float(r[key])
            for r in wrong_rows
        ]

        print(label)

        print(
            f"  correct mean = "
            f"{mean(c):.4f}"
        )

        print(
            f"  wrong mean   = "
            f"{mean(w):.4f}"
        )

        print(
            f"  correct range = "
            f"[{min(c):.4f}, {max(c):.4f}]"
        )

        print(
            f"  wrong range   = "
            f"[{min(w):.4f}, {max(w):.4f}]"
        )

        print()

    print(
        "No confidence-routing threshold is selected."
    )

    print(
        "Detector confidence is treated only as "
        "observation reliability, NOT identity confidence."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
