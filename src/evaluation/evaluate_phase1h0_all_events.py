import csv
from pathlib import Path
from statistics import mean

import torch
from PIL import Image
from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder
from recovery.annotation_candidate_generator import get_frame_candidates
from recovery.real_candidate_generator import filter_real_candidates


ROOT = Path(__file__).resolve().parents[2]

ELIGIBLE_FILE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h0_all_events.csv"
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


def main():

    encoder = AppearanceEncoder()

    detector = YOLO("yolo11n.pt")

    with ELIGIBLE_FILE.open(newline="") as f:
        eligible = list(csv.DictReader(f))

    rows = []

    print(
        "PHASE 1H-0 — ALL-EVENT BASELINE IDENTITY MATCHING"
    )
    print("=" * 92)

    for item in eligible:

        event_id = item["event_id"]

        event = load_event(event_id)

        sequence = event["sequence"]
        target_id = int(event["target_id"])

        selection_frame = int(
            event["selection_frame"]
        )

        reappearance_frame = int(
            event["reappearance_frame"]
        )

        reference_box = get_xyxy(
            event["selection_box"]
        )

        # --------------------------------------------------
        # Original user-selected reference crop
        # --------------------------------------------------

        reference_path = (
            IMAGE_ROOT
            / sequence
            / f"img{selection_frame:05d}.jpg"
        )

        reference_image = Image.open(
            reference_path
        ).convert("RGB")

        reference_box_int = clamp_box(
            reference_box,
            reference_image.width,
            reference_image.height,
        )

        reference_crop = reference_image.crop(
            reference_box_int
        )

        reference_embedding = encoder.encode(
            reference_crop
        )

        # --------------------------------------------------
        # Frozen Phase 1G candidates
        # --------------------------------------------------

        candidate_path = (
            IMAGE_ROOT
            / sequence
            / f"img{reappearance_frame:05d}.jpg"
        )

        candidate_image = Image.open(
            candidate_path
        ).convert("RGB")

        result = detector.predict(
            source=str(candidate_path),
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
        # Rank candidates WITHOUT GT
        # --------------------------------------------------

        scored = []

        for candidate in candidates:

            box = clamp_box(
                candidate["box_xyxy"],
                candidate_image.width,
                candidate_image.height,
            )

            x1, y1, x2, y2 = box

            if x2 <= x1 or y2 <= y1:
                continue

            crop = candidate_image.crop(box)

            embedding = encoder.encode(crop)

            similarity = torch.dot(
                reference_embedding,
                embedding,
            ).item()

            scored.append({
                "box_xyxy": box,
                "similarity": similarity,
                "detector_confidence":
                    candidate["confidence"],
            })

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
        # Ground truth enters ONLY AFTER ranking
        # --------------------------------------------------

        xml_path = (
            XML_ROOT
            / f"{sequence}.xml"
        )

        gt_objects = get_frame_candidates(
            str(xml_path),
            reappearance_frame,
        )

        target_gt = next(
            g for g in gt_objects
            if int(g["target_id"]) == target_id
        )

        target_gt_box = target_gt["box_xyxy"]

        for candidate in scored:

            candidate["target_iou"] = iou(
                candidate["box_xyxy"],
                target_gt_box,
            )

            candidate["is_target"] = (
                candidate["target_iou"]
                >= TARGET_IOU_THRESHOLD
            )

        target_candidates = [
            c for c in scored
            if c["is_target"]
        ]

        distractors = [
            c for c in scored
            if not c["is_target"]
        ]

        target_rank = (
            min(
                c["rank"]
                for c in target_candidates
            )
            if target_candidates
            else None
        )

        target_similarity = (
            max(
                c["similarity"]
                for c in target_candidates
            )
            if target_candidates
            else None
        )

        hardest_distractor_similarity = (
            max(
                c["similarity"]
                for c in distractors
            )
            if distractors
            else None
        )

        target_margin = None

        if (
            target_similarity is not None
            and
            hardest_distractor_similarity
            is not None
        ):
            target_margin = (
                target_similarity
                - hardest_distractor_similarity
            )

        top1_top2_margin = None

        if len(scored) >= 2:
            top1_top2_margin = (
                scored[0]["similarity"]
                - scored[1]["similarity"]
            )

        rank1 = (
            target_rank == 1
        )

        rank3 = (
            target_rank is not None
            and target_rank <= 3
        )

        reciprocal_rank = (
            1.0 / target_rank
            if target_rank is not None
            else 0.0
        )

        rows.append({
            "event_id":
                event_id,

            "sequence":
                sequence,

            "selection_frame":
                selection_frame,

            "reappearance_frame":
                reappearance_frame,

            "raw_candidate_count":
                len(raw_candidates),

            "filtered_candidate_count":
                len(scored),

            "target_found_in_candidates":
                bool(target_candidates),

            "target_rank":
                target_rank
                if target_rank is not None
                else "",

            "rank1_correct":
                rank1,

            "rank3_correct":
                rank3,

            "reciprocal_rank":
                round(
                    reciprocal_rank,
                    6,
                ),

            "target_similarity":
                (
                    round(
                        target_similarity,
                        6,
                    )
                    if target_similarity
                    is not None
                    else ""
                ),

            "hardest_distractor_similarity":
                (
                    round(
                        hardest_distractor_similarity,
                        6,
                    )
                    if hardest_distractor_similarity
                    is not None
                    else ""
                ),

            "target_vs_hardest_distractor_margin":
                (
                    round(
                        target_margin,
                        6,
                    )
                    if target_margin
                    is not None
                    else ""
                ),

            "top1_top2_margin":
                (
                    round(
                        top1_top2_margin,
                        6,
                    )
                    if top1_top2_margin
                    is not None
                    else ""
                ),
        })

        print(
            f"{event_id} | "
            f"candidates={len(scored):2d} | "
            f"target_rank={str(target_rank):>2} | "
            f"R1={str(rank1):5} | "
            f"R3={str(rank3):5} | "
            f"target_sim="
            f"{target_similarity:.4f} | "
            f"hardest="
            f"{hardest_distractor_similarity:.4f} | "
            f"margin="
            f"{target_margin:+.4f}"
        )

    # --------------------------------------------------
    # Save event-level results
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
    # Overall summary
    # --------------------------------------------------

    n = len(rows)

    rank1_hits = sum(
        row["rank1_correct"]
        for row in rows
    )

    rank3_hits = sum(
        row["rank3_correct"]
        for row in rows
    )

    candidate_hits = sum(
        row["target_found_in_candidates"]
        for row in rows
    )

    reciprocal_ranks = [
        float(row["reciprocal_rank"])
        for row in rows
    ]

    numeric_ranks = [
        int(row["target_rank"])
        for row in rows
        if str(row["target_rank"]).strip()
    ]

    margins = [
        float(
            row[
                "target_vs_hardest_distractor_margin"
            ]
        )
        for row in rows
        if str(
            row[
                "target_vs_hardest_distractor_margin"
            ]
        ).strip()
    ]

    positive_margins = sum(
        margin > 0
        for margin in margins
    )

    print()
    print("-" * 92)

    print(
        f"Target present in candidate set: "
        f"{candidate_hits}/{n}"
    )

    print(
        f"Rank-1 accuracy: "
        f"{rank1_hits}/{n} "
        f"= {rank1_hits/n:.3f}"
    )

    print(
        f"Rank-3 accuracy: "
        f"{rank3_hits}/{n} "
        f"= {rank3_hits/n:.3f}"
    )

    print(
        f"Mean reciprocal rank: "
        f"{mean(reciprocal_ranks):.3f}"
    )

    if numeric_ranks:
        print(
            f"Mean target rank: "
            f"{mean(numeric_ranks):.2f}"
        )

    print(
        f"Positive target-vs-distractor margin: "
        f"{positive_margins}/{len(margins)}"
    )

    if margins:
        print(
            f"Mean target-vs-distractor margin: "
            f"{mean(margins):+.4f}"
        )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
