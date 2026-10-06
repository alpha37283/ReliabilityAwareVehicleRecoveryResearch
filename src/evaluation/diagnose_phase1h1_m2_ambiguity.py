import csv
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder
from recovery.recent_memory import load_recent_memory
from recovery.distractor_memory import build_distractor_memory
from recovery.annotation_candidate_generator import get_frame_candidates
from recovery.real_candidate_generator import filter_real_candidates


ROOT = Path(__file__).resolve().parents[2]

ELIGIBLE = ROOT / "results/phase1/phase1e_candidate_recall.csv"

IMAGE_ROOT = ROOT / "data/raw/UA-DETRAC/images"

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h1_m2_ambiguity_diagnostic.csv"
)

TARGET_IOU = 0.30


def get_xyxy(box):
    return box["xyxy"] if isinstance(box, dict) else box


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box

    return [
        int(max(0, min(w, round(float(x1))))),
        int(max(0, min(h, round(float(y1))))),
        int(max(0, min(w, round(float(x2))))),
        int(max(0, min(h, round(float(y2))))),
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

    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def main():

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    rows = []

    print("PHASE 1H-1 — DISTRACTOR-MEMORY AMBIGUITY DIAGNOSTIC")
    print("=" * 112)

    for item in events:

        event_id = item["event_id"]
        event = load_event(event_id)

        sequence = event["sequence"]
        target_id = int(event["target_id"])

        selection_frame = int(event["selection_frame"])
        occlusion_start = int(event["occlusion_start"])
        reappearance = int(event["reappearance_frame"])

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
        # Same M1 positive memory
        # --------------------------------------------------

        positive_memory = load_recent_memory(
            predictions_csv=predictions_csv,
            image_dir=IMAGE_ROOT / sequence,
            selection_frame=selection_frame,
            memory_cutoff_frame=occlusion_start,
            selection_box=selection_box,
            capacity=5,
        )

        positive_embeddings = torch.stack([
            encoder.encode(x["crop"])
            for x in positive_memory
        ])

        # --------------------------------------------------
        # M2 negative distractor memory
        # --------------------------------------------------

        negative_memory = build_distractor_memory(
            predictions_csv=predictions_csv,
            image_dir=IMAGE_ROOT / sequence,
            selection_frame=selection_frame,
            memory_cutoff_frame=occlusion_start,
            encoder=encoder,
            detector=detector,
            capacity=3,
        )

        if not negative_memory:
            continue

        negative_embeddings = torch.stack([
            x["embedding"]
            for x in negative_memory
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

            cls = int(box.cls[0])

            raw.append({
                "box_xyxy": box.xyxy[0].tolist(),
                "confidence": float(box.conf[0]),
                "class_id": cls,
                "class_name": result.names[cls],
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

            emb = encoder.encode(crop)

            positive = float(
                torch.max(
                    torch.mv(
                        positive_embeddings,
                        emb,
                    )
                ).item()
            )

            negative = float(
                torch.max(
                    torch.mv(
                        negative_embeddings,
                        emb,
                    )
                ).item()
            )

            scored.append({
                "box_xyxy": box,
                "positive_score": positive,
                "negative_score": negative,
                "ambiguity_gap": positive - negative,
            })

        # IMPORTANT:
        # Ranking stays M1: positive evidence only.
        scored.sort(
            key=lambda x: x["positive_score"],
            reverse=True,
        )

        top1 = scored[0]

        top2_positive = (
            scored[1]["positive_score"]
            if len(scored) > 1
            else None
        )

        top1_top2_margin = (
            top1["positive_score"]
            - top2_positive
            if top2_positive is not None
            else None
        )

        # --------------------------------------------------
        # GT only after ranking
        # --------------------------------------------------

        xml_path = XML_ROOT / f"{sequence}.xml"

        gt = get_frame_candidates(
            str(xml_path),
            reappearance,
        )

        target_gt = next(
            x for x in gt
            if int(x["target_id"]) == target_id
        )

        target_iou = iou(
            top1["box_xyxy"],
            target_gt["box_xyxy"],
        )

        correct = target_iou >= TARGET_IOU

        row = {
            "event_id": event_id,
            "negative_memory_size": len(negative_memory),
            "top1_correct": correct,

            "top1_positive_score":
                round(top1["positive_score"], 6),

            "top1_negative_score":
                round(top1["negative_score"], 6),

            "top1_ambiguity_gap":
                round(top1["ambiguity_gap"], 6),

            "top1_top2_positive_margin":
                round(top1_top2_margin, 6)
                if top1_top2_margin is not None
                else "",

            "top1_target_iou":
                round(target_iou, 4),
        }

        rows.append(row)

        print(
            f"{event_id} | "
            f"correct={str(correct):5} | "
            f"pos={top1['positive_score']:.4f} | "
            f"neg={top1['negative_score']:.4f} | "
            f"gap={top1['ambiguity_gap']:+.4f} | "
            f"top1-top2="
            f"{top1_top2_margin:+.4f}"
        )

    with OUTPUT.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("-" * 112)

    correct_rows = [
        r for r in rows
        if r["top1_correct"]
    ]

    wrong_rows = [
        r for r in rows
        if not r["top1_correct"]
    ]

    print(
        f"Active negative-memory events: {len(rows)}"
    )

    print(
        f"M1 Rank-1 correct: "
        f"{len(correct_rows)}/{len(rows)}"
    )

    print(
        f"M1 Rank-1 wrong: "
        f"{len(wrong_rows)}/{len(rows)}"
    )

    if correct_rows:
        print(
            "Correct top1 ambiguity gaps: "
            + ", ".join(
                f"{float(r['top1_ambiguity_gap']):+.4f}"
                for r in correct_rows
            )
        )

    if wrong_rows:
        print(
            "Wrong top1 ambiguity gaps:   "
            + ", ".join(
                f"{float(r['top1_ambiguity_gap']):+.4f}"
                for r in wrong_rows
            )
        )

    print()
    print(
        "No ambiguity threshold is chosen in this diagnostic."
    )

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
