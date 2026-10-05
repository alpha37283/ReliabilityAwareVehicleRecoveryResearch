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

ELIGIBLE_FILE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

H0_FILE = (
    ROOT
    / "results/phase1/phase1h0_all_events.csv"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h1_m1_all_events.csv"
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
MEMORY_CAPACITY = 5


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

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def main():

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    with ELIGIBLE_FILE.open(newline="") as f:
        eligible = list(csv.DictReader(f))

    with H0_FILE.open(newline="") as f:
        h0_rows = {
            row["event_id"]: row
            for row in csv.DictReader(f)
        }

    rows = []

    print(
        "PHASE 1H-1 — M1 RECENT-MEMORY IDENTITY MATCHING"
    )
    print("=" * 106)

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
        
        memory_cutoff_frame=int(
            event["occlusion_start"]
        ),

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
        # Build M1 memory WITHOUT GT
        # --------------------------------------------------

        memory = load_recent_memory(
            predictions_csv=predictions_csv,
            image_dir=IMAGE_ROOT / sequence,
            selection_frame=selection_frame,
            memory_cutoff_frame=int(
                event["occlusion_start"]
            ),
            selection_box=selection_box,
            capacity=MEMORY_CAPACITY,
        )

        memory_embeddings = []

        for entry in memory:

            emb = encoder.encode(
                entry["crop"]
            )

            memory_embeddings.append(
                emb
            )

        memory_matrix = torch.stack(
            memory_embeddings
        )

        # --------------------------------------------------
        # Frozen Phase 1G candidate generation
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
            selection_box,
        )

        # --------------------------------------------------
        # M1 candidate scoring
        #
        # candidate score =
        # maximum cosine similarity to ANY memory entry.
        #
        # No GT is used here.
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

            candidate_embedding = encoder.encode(
                crop
            )

            similarities = torch.mv(
                memory_matrix,
                candidate_embedding,
            )

            best_memory_index = int(
                torch.argmax(
                    similarities
                ).item()
            )

            score = float(
                similarities[
                    best_memory_index
                ].item()
            )

            best_memory = memory[
                best_memory_index
            ]

            scored.append({
                "box_xyxy":
                    box,

                "score":
                    score,

                "detector_confidence":
                    candidate["confidence"],

                "best_memory_frame":
                    best_memory[
                        "frame_number"
                    ],

                "best_memory_source":
                    best_memory[
                        "source"
                    ],
            })

        scored.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        for rank, candidate in enumerate(
            scored,
            start=1,
        ):
            candidate["rank"] = rank

        # --------------------------------------------------
        # GT enters ONLY AFTER ranking
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

        target_gt_box = target_gt[
            "box_xyxy"
        ]

        for candidate in scored:

            target_iou = iou(
                candidate["box_xyxy"],
                target_gt_box,
            )

            candidate["target_iou"] = (
                target_iou
            )

            candidate["is_target"] = (
                target_iou
                >= TARGET_IOU_THRESHOLD
            )

        targets = [
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
                for c in targets
            )
            if targets
            else None
        )

        target_best = (
            max(
                targets,
                key=lambda x: x["score"],
            )
            if targets
            else None
        )

        hardest_distractor = (
            max(
                distractors,
                key=lambda x: x["score"],
            )
            if distractors
            else None
        )

        target_similarity = (
            target_best["score"]
            if target_best
            else None
        )

        hardest_similarity = (
            hardest_distractor["score"]
            if hardest_distractor
            else None
        )

        margin = None

        if (
            target_similarity is not None
            and hardest_similarity is not None
        ):
            margin = (
                target_similarity
                - hardest_similarity
            )

        rank1 = target_rank == 1

        rank3 = (
            target_rank is not None
            and target_rank <= 3
        )

        reciprocal_rank = (
            1.0 / target_rank
            if target_rank is not None
            else 0.0
        )

        h0 = h0_rows[event_id]

        h0_rank = int(
            h0["target_rank"]
        )

        rank_change = (
            h0_rank - target_rank
            if target_rank is not None
            else ""
        )

        rows.append({
            "event_id":
                event_id,

            "memory_size":
                len(memory),

            "memory_frames":
                ",".join(
                    str(x["frame_number"])
                    for x in memory
                ),

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

            "target_best_memory_frame":
                (
                    target_best[
                        "best_memory_frame"
                    ]
                    if target_best
                    else ""
                ),

            "target_best_memory_source":
                (
                    target_best[
                        "best_memory_source"
                    ]
                    if target_best
                    else ""
                ),

            "hardest_distractor_similarity":
                (
                    round(
                        hardest_similarity,
                        6,
                    )
                    if hardest_similarity
                    is not None
                    else ""
                ),

            "hardest_distractor_best_memory_frame":
                (
                    hardest_distractor[
                        "best_memory_frame"
                    ]
                    if hardest_distractor
                    else ""
                ),

            "target_vs_hardest_distractor_margin":
                (
                    round(
                        margin,
                        6,
                    )
                    if margin is not None
                    else ""
                ),

            "h0_target_rank":
                h0_rank,

            "rank_improvement_vs_h0":
                rank_change,
        })

        print(
            f"{event_id} | "
            f"H0rank={h0_rank:2d} -> "
            f"M1rank={target_rank:2d} | "
            f"R1={str(rank1):5} | "
            f"R3={str(rank3):5} | "
            f"target={target_similarity:.4f} | "
            f"hardest={hardest_similarity:.4f} | "
            f"margin={margin:+.4f} | "
            f"target_best_mem="
            f"{target_best['best_memory_frame']}"
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
    # Summary
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

    ranks = [
        int(row["target_rank"])
        for row in rows
        if str(
            row["target_rank"]
        ).strip()
    ]

    rr = [
        float(row["reciprocal_rank"])
        for row in rows
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

    positive = sum(
        x > 0
        for x in margins
    )

    improved = sum(
        int(row["rank_improvement_vs_h0"]) > 0
        for row in rows
    )

    unchanged = sum(
        int(row["rank_improvement_vs_h0"]) == 0
        for row in rows
    )

    worsened = sum(
        int(row["rank_improvement_vs_h0"]) < 0
        for row in rows
    )

    print()
    print("-" * 106)

    print(
        f"M1 Rank-1 accuracy: "
        f"{rank1_hits}/{n} "
        f"= {rank1_hits/n:.3f}"
    )

    print(
        f"M1 Rank-3 accuracy: "
        f"{rank3_hits}/{n} "
        f"= {rank3_hits/n:.3f}"
    )

    print(
        f"M1 mean target rank: "
        f"{mean(ranks):.2f}"
    )

    print(
        f"M1 mean reciprocal rank: "
        f"{mean(rr):.3f}"
    )

    print(
        f"M1 positive target margin: "
        f"{positive}/{len(margins)}"
    )

    print(
        f"M1 mean target margin: "
        f"{mean(margins):+.4f}"
    )

    print()
    print(
        f"Rank improved vs H0:  "
        f"{improved}/{n}"
    )

    print(
        f"Rank unchanged vs H0: "
        f"{unchanged}/{n}"
    )

    print(
        f"Rank worsened vs H0:  "
        f"{worsened}/{n}"
    )

    print()
    print(
        "H0 reference: "
        "Rank-1=5/10, Rank-3=6/10, "
        "mean rank=4.70, MRR=0.585"
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
