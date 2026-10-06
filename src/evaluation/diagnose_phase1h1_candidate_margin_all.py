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

OUTPUT = (
    ROOT
    / "results/phase1/phase1h1_candidate_margin_all.csv"
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

TARGET_IOU_THRESHOLD = 0.30


def get_xyxy(box):
    if isinstance(box, dict):
        return box["xyxy"]
    return box


def clamp_box(box, width, height):

    x1, y1, x2, y2 = box

    return [
        int(max(0, min(width, round(float(x1))))),
        int(max(0, min(height, round(float(y1))))),
        int(max(0, min(width, round(float(x2))))),
        int(max(0, min(height, round(float(y2))))),
    ]


def iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = (
        max(0.0, ix2 - ix1)
        * max(0.0, iy2 - iy1)
    )

    area_a = (
        max(0.0, ax2-ax1)
        * max(0.0, ay2-ay1)
    )

    area_b = (
        max(0.0, bx2-bx1)
        * max(0.0, by2-by1)
    )

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def main():

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    rows = []

    print(
        "PHASE 1H-1 — ALL-EVENT M1 CANDIDATE-MARGIN DIAGNOSTIC"
    )
    print("=" * 110)

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

        reappearance_frame = int(
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
        # Same frozen M1 positive memory
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
            / f"img{reappearance_frame:05d}.jpg"
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

            cls = int(box.cls[0])

            raw.append({
                "box_xyxy":
                    box.xyxy[0].tolist(),

                "confidence":
                    float(box.conf[0]),

                "class_id":
                    cls,

                "class_name":
                    result.names[cls],
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

            score = float(
                torch.max(
                    torch.mv(
                        memory_embeddings,
                        embedding,
                    )
                ).item()
            )

            scored.append({
                "box_xyxy": box,
                "score": score,
            })

        scored.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        if len(scored) < 2:
            print(
                f"{event_id} | fewer than 2 candidates"
            )
            continue

        top1 = scored[0]
        top2 = scored[1]

        margin = (
            top1["score"]
            - top2["score"]
        )

        # --------------------------------------------------
        # GT only AFTER ranking
        # --------------------------------------------------

        xml_path = (
            XML_ROOT
            / f"{sequence}.xml"
        )

        gt = get_frame_candidates(
            str(xml_path),
            reappearance_frame,
        )

        target_gt = next(
            x for x in gt
            if int(x["target_id"]) == target_id
        )

        target_iou = iou(
            top1["box_xyxy"],
            target_gt["box_xyxy"],
        )

        correct = (
            target_iou
            >= TARGET_IOU_THRESHOLD
        )

        rows.append({
            "event_id":
                event_id,

            "memory_size":
                len(memory),

            "candidate_count":
                len(scored),

            "top1_score":
                round(top1["score"], 6),

            "top2_score":
                round(top2["score"], 6),

            "top1_top2_margin":
                round(margin, 6),

            "top1_correct":
                correct,

            "top1_target_iou":
                round(target_iou, 4),
        })

        print(
            f"{event_id} | "
            f"correct={str(correct):5} | "
            f"top1={top1['score']:.4f} | "
            f"top2={top2['score']:.4f} | "
            f"margin={margin:+.4f} | "
            f"candidates={len(scored)}"
        )

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    correct_rows = [
        r for r in rows
        if r["top1_correct"]
    ]

    wrong_rows = [
        r for r in rows
        if not r["top1_correct"]
    ]

    print()
    print("-" * 110)

    print(
        f"Correct Rank-1 decisions: "
        f"{len(correct_rows)}/{len(rows)}"
    )

    print(
        f"Wrong Rank-1 decisions: "
        f"{len(wrong_rows)}/{len(rows)}"
    )

    if correct_rows:

        correct_margins = [
            float(r["top1_top2_margin"])
            for r in correct_rows
        ]

        print(
            "Correct margins: "
            + ", ".join(
                f"{x:+.4f}"
                for x in correct_margins
            )
        )

        print(
            f"Mean correct margin: "
            f"{mean(correct_margins):+.4f}"
        )

        print(
            f"Minimum correct margin: "
            f"{min(correct_margins):+.4f}"
        )

    if wrong_rows:

        wrong_margins = [
            float(r["top1_top2_margin"])
            for r in wrong_rows
        ]

        print(
            "Wrong margins:   "
            + ", ".join(
                f"{x:+.4f}"
                for x in wrong_margins
            )
        )

        print(
            f"Mean wrong margin: "
            f"{mean(wrong_margins):+.4f}"
        )

        print(
            f"Maximum wrong margin: "
            f"{max(wrong_margins):+.4f}"
        )

    print()
    print(
        "No MATCH / NEED-MORE-EVIDENCE threshold "
        "is selected in this diagnostic."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
