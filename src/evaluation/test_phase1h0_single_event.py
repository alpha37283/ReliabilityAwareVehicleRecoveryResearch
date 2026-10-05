import csv
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder
from recovery.annotation_candidate_generator import get_frame_candidates
from recovery.real_candidate_generator import filter_real_candidates


ROOT = Path(__file__).resolve().parents[2]

EVENT_ID = "E011"

IMAGE_ROOT = ROOT / "data/raw/UA-DETRAC/images"

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

OUTPUT_DIR = (
    ROOT
    / "results/phase1/phase1h0_debug"
)

IMGSZ = 640
CONF = 0.20
TARGET_IOU_THRESHOLD = 0.30


def get_xyxy(box):
    if isinstance(box, dict):
        return box["xyxy"]

    return box


def clamp_box(box, width, height):

    x1, y1, x2, y2 = box

    return [
        int(max(0, min(width, round(x1)))),
        int(max(0, min(height, round(y1)))),
        int(max(0, min(width, round(x2)))),
        int(max(0, min(height, round(y2)))),
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

    intersection = iw * ih

    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    event = load_event(EVENT_ID)

    sequence = event["sequence"]
    target_id = int(event["target_id"])

    selection_frame = int(
        event["selection_frame"]
    )

    candidate_frame = int(
        event["reappearance_frame"]
    )

    reference_box = get_xyxy(
        event["selection_box"]
    )

    # --------------------------------------------------
    # Load reference image/crop
    # --------------------------------------------------

    reference_image_path = (
        IMAGE_ROOT
        / sequence
        / f"img{selection_frame:05d}.jpg"
    )

    reference_image = Image.open(
        reference_image_path
    ).convert("RGB")

    reference_box_int = clamp_box(
        reference_box,
        reference_image.width,
        reference_image.height,
    )

    reference_crop = reference_image.crop(
        reference_box_int
    )

    # --------------------------------------------------
    # Appearance encoder
    # --------------------------------------------------

    encoder = AppearanceEncoder()

    reference_embedding = encoder.encode(
        reference_crop
    )

    # --------------------------------------------------
    # Run frozen Phase 1G detector
    # --------------------------------------------------

    candidate_image_path = (
        IMAGE_ROOT
        / sequence
        / f"img{candidate_frame:05d}.jpg"
    )

    candidate_image = Image.open(
        candidate_image_path
    ).convert("RGB")

    detector = YOLO("yolo11n.pt")

    result = detector.predict(
        source=str(candidate_image_path),
        imgsz=IMGSZ,
        conf=CONF,
        device=0,
        classes=[2, 3, 5, 7],
        verbose=False,
    )[0]

    raw_candidates = []

    for box in result.boxes:

        class_id = int(box.cls[0])

        raw_candidates.append({
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
        raw_candidates,
        reference_box,
    )

    # --------------------------------------------------
    # Score candidates WITHOUT ground truth
    # --------------------------------------------------

    scored = []

    for index, candidate in enumerate(candidates):

        candidate_box = clamp_box(
            candidate["box_xyxy"],
            candidate_image.width,
            candidate_image.height,
        )

        x1, y1, x2, y2 = candidate_box

        if x2 <= x1 or y2 <= y1:
            continue

        crop = candidate_image.crop(
            candidate_box
        )

        embedding = encoder.encode(crop)

        # Both embeddings are L2 normalized,
        # therefore dot product == cosine similarity.
        similarity = torch.dot(
            reference_embedding,
            embedding,
        ).item()

        scored.append({
            "candidate_index": index,
            "box_xyxy": candidate_box,
            "detector_confidence":
                candidate["confidence"],
            "class_name":
                candidate["class_name"],
            "similarity":
                similarity,
        })

    # Rank purely using visual similarity.
    scored.sort(
        key=lambda x: x["similarity"],
        reverse=True,
    )

    for rank, candidate in enumerate(
        scored,
        start=1,
    ):
        candidate["rank"] = rank

    # --------------------------------------------------
    # Ground truth enters ONLY NOW for evaluation.
    # --------------------------------------------------

    xml_path = XML_ROOT / f"{sequence}.xml"

    gt_objects = get_frame_candidates(
        str(xml_path),
        candidate_frame,
    )

    target_gt = next(
        g for g in gt_objects
        if int(g["target_id"]) == target_id
    )

    target_gt_box = target_gt["box_xyxy"]

    for candidate in scored:

        target_iou = iou(
            candidate["box_xyxy"],
            target_gt_box,
        )

        candidate["target_iou"] = target_iou

        candidate["is_target_candidate"] = (
            target_iou >= TARGET_IOU_THRESHOLD
        )

    target_matches = [
        c for c in scored
        if c["is_target_candidate"]
    ]

    target_rank = (
        min(c["rank"] for c in target_matches)
        if target_matches
        else None
    )

    target_similarity = (
        max(
            c["similarity"]
            for c in target_matches
        )
        if target_matches
        else None
    )

    distractors = [
        c for c in scored
        if not c["is_target_candidate"]
    ]

    hardest_distractor_similarity = (
        max(
            c["similarity"]
            for c in distractors
        )
        if distractors
        else None
    )

    top1_top2_margin = None

    if len(scored) >= 2:
        top1_top2_margin = (
            scored[0]["similarity"]
            - scored[1]["similarity"]
        )

    target_margin = None

    if (
        target_similarity is not None
        and hardest_distractor_similarity
        is not None
    ):
        target_margin = (
            target_similarity
            - hardest_distractor_similarity
        )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / f"{EVENT_ID}_candidate_ranking.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "candidate_index",
                "class_name",
                "detector_confidence",
                "similarity",
                "target_iou",
                "is_target_candidate",
                "box_xyxy",
            ],
        )

        writer.writeheader()

        for candidate in scored:

            row = candidate.copy()

            row["detector_confidence"] = round(
                row["detector_confidence"],
                4,
            )

            row["similarity"] = round(
                row["similarity"],
                6,
            )

            row["target_iou"] = round(
                row["target_iou"],
                4,
            )

            writer.writerow(row)

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    debug_image = candidate_image.copy()

    draw = ImageDraw.Draw(debug_image)

    for candidate in scored:

        x1, y1, x2, y2 = candidate["box_xyxy"]

        color = (
            "lime"
            if candidate["is_target_candidate"]
            else "red"
        )

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=color,
            width=2,
        )

        label = (
            f"#{candidate['rank']} "
            f"{candidate['similarity']:.3f}"
        )

        draw.text(
            (x1, max(0, y1 - 12)),
            label,
            fill=color,
        )

    visualization_path = (
        OUTPUT_DIR
        / f"{EVENT_ID}_candidate_ranking.jpg"
    )

    debug_image.save(
        visualization_path
    )

    # --------------------------------------------------
    # Terminal summary
    # --------------------------------------------------

    print(
        "PHASE 1H-0 — SINGLE-EVENT IDENTITY MATCHING"
    )
    print("=" * 72)

    print(f"Event:                       {EVENT_ID}")
    print(f"Selection frame:             {selection_frame}")
    print(f"Candidate frame:             {candidate_frame}")

    print(
        f"Raw Phase 1G candidates:     "
        f"{len(raw_candidates)}"
    )

    print(
        f"Filtered candidates:         "
        f"{len(scored)}"
    )

    print()
    print("RANKING")
    print("-" * 72)

    for candidate in scored:

        marker = (
            "<-- TARGET"
            if candidate["is_target_candidate"]
            else ""
        )

        print(
            f"Rank {candidate['rank']:2d} | "
            f"sim={candidate['similarity']:.4f} | "
            f"det={candidate['detector_confidence']:.3f} | "
            f"targetIoU={candidate['target_iou']:.3f} "
            f"{marker}"
        )

    print()
    print("-" * 72)

    print(
        f"Target rank:                 "
        f"{target_rank}"
    )

    print(
        f"Rank-1 correct:              "
        f"{target_rank == 1}"
    )

    print(
        f"Rank-3 correct:              "
        f"{target_rank is not None and target_rank <= 3}"
    )

    print(
        f"Target similarity:           "
        f"{target_similarity}"
    )

    print(
        f"Hardest distractor similarity:"
        f" {hardest_distractor_similarity}"
    )

    print(
        f"Target-vs-distractor margin: "
        f"{target_margin}"
    )

    print(
        f"Top1-top2 margin:            "
        f"{top1_top2_margin}"
    )

    print()
    print(f"CSV:           {csv_path}")
    print(f"Visualization: {visualization_path}")


if __name__ == "__main__":
    main()
