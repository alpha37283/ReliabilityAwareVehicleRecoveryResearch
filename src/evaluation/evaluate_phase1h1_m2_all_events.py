import csv
from pathlib import Path
from statistics import mean

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

ELIGIBLE_FILE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

H0_FILE = (
    ROOT
    / "results/phase1/phase1h0_all_events.csv"
)

M1_FILE = (
    ROOT
    / "results/phase1/phase1h1_m1_all_events.csv"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h1_m2_all_events.csv"
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

POSITIVE_MEMORY_CAPACITY = 5
NEGATIVE_MEMORY_CAPACITY = 3


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


def read_lookup(path):

    with path.open(newline="") as f:
        return {
            row["event_id"]: row
            for row in csv.DictReader(f)
        }


def main():

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    h0_lookup = read_lookup(H0_FILE)
    m1_lookup = read_lookup(M1_FILE)

    with ELIGIBLE_FILE.open(newline="") as f:
        events = list(csv.DictReader(f))

    rows = []

    print(
        "PHASE 1H-1 — M2 DISTRACTOR-AWARE IDENTITY MATCHING"
    )
    print("=" * 118)

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
        # Positive memory = EXACTLY M1
        # --------------------------------------------------

        positive_memory = load_recent_memory(
            predictions_csv=predictions_csv,
            image_dir=IMAGE_ROOT / sequence,
            selection_frame=selection_frame,
            memory_cutoff_frame=occlusion_start,
            selection_box=selection_box,
            capacity=POSITIVE_MEMORY_CAPACITY,
        )

        positive_embeddings = torch.stack([
            encoder.encode(entry["crop"])
            for entry in positive_memory
        ])

        # --------------------------------------------------
        # Negative distractor memory
        # No GT involved.
        # --------------------------------------------------

        negative_memory = build_distractor_memory(
            predictions_csv=predictions_csv,
            image_dir=IMAGE_ROOT / sequence,
            selection_frame=selection_frame,
            memory_cutoff_frame=occlusion_start,
            encoder=encoder,
            detector=detector,
            capacity=NEGATIVE_MEMORY_CAPACITY,
        )

        if negative_memory:

            negative_embeddings = torch.stack([
                entry["embedding"]
                for entry in negative_memory
            ])

        else:
            negative_embeddings = None

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
            selection_box,
        )

        # --------------------------------------------------
        # M2 scoring — NO GT
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

            positive_similarities = torch.mv(
                positive_embeddings,
                candidate_embedding,
            )

            positive_score = float(
                torch.max(
                    positive_similarities
                ).item()
            )

            if negative_embeddings is not None:

                negative_similarities = torch.mv(
                    negative_embeddings,
                    candidate_embedding,
                )

                negative_score = float(
                    torch.max(
                        negative_similarities
                    ).item()
                )

                final_score = (
                    positive_score
                    - negative_score
                )

            else:

                negative_score = None

                # M2 falls back exactly to M1.
                final_score = positive_score

            scored.append({
                "box_xyxy":
                    box,

                "positive_score":
                    positive_score,

                "negative_score":
                    negative_score,

                "final_score":
                    final_score,
            })

        scored.sort(
            key=lambda x:
                x["final_score"],
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

        target_gt_box = target_gt[
            "box_xyxy"
        ]

        for candidate in scored:

            candidate["target_iou"] = iou(
                candidate["box_xyxy"],
                target_gt_box,
            )

            candidate["is_target"] = (
                candidate["target_iou"]
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
                key=lambda x:
                    x["final_score"],
            )
            if targets
            else None
        )

        hardest_distractor = (
            max(
                distractors,
                key=lambda x:
                    x["final_score"],
            )
            if distractors
            else None
        )

        target_score = (
            target_best["final_score"]
            if target_best
            else None
        )

        hardest_score = (
            hardest_distractor["final_score"]
            if hardest_distractor
            else None
        )

        margin = None

        if (
            target_score is not None
            and hardest_score is not None
        ):
            margin = (
                target_score
                - hardest_score
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

        h0_rank = int(
            h0_lookup[event_id][
                "target_rank"
            ]
        )

        m1_rank = int(
            m1_lookup[event_id][
                "target_rank"
            ]
        )

        rows.append({
            "event_id":
                event_id,

            "positive_memory_size":
                len(positive_memory),

            "negative_memory_size":
                len(negative_memory),

            "target_rank":
                target_rank,

            "rank1_correct":
                rank1,

            "rank3_correct":
                rank3,

            "reciprocal_rank":
                round(
                    reciprocal_rank,
                    6,
                ),

            "target_positive_score":
                round(
                    target_best[
                        "positive_score"
                    ],
                    6,
                )
                if target_best else "",

            "target_negative_score":
                (
                    round(
                        target_best[
                            "negative_score"
                        ],
                        6,
                    )
                    if (
                        target_best
                        and
                        target_best[
                            "negative_score"
                        ] is not None
                    )
                    else ""
                ),

            "target_final_score":
                round(
                    target_score,
                    6,
                )
                if target_score
                is not None
                else "",

            "hardest_distractor_score":
                round(
                    hardest_score,
                    6,
                )
                if hardest_score
                is not None
                else "",

            "target_vs_hardest_distractor_margin":
                round(
                    margin,
                    6,
                )
                if margin is not None
                else "",

            "h0_rank":
                h0_rank,

            "m1_rank":
                m1_rank,

            "m2_rank_change_vs_m1":
                (
                    m1_rank
                    - target_rank
                    if target_rank is not None
                    else ""
                ),
        })

        print(
            f"{event_id} | "
            f"negmem={len(negative_memory)} | "
            f"H0={h0_rank:2d} | "
            f"M1={m1_rank:2d} | "
            f"M2={target_rank:2d} | "
            f"R1={str(rank1):5} | "
            f"R3={str(rank3):5} | "
            f"target={target_score:+.4f} | "
            f"hardest={hardest_score:+.4f} | "
            f"margin={margin:+.4f}"
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
    # Overall summary
    # --------------------------------------------------

    n = len(rows)

    rank1_hits = sum(
        r["rank1_correct"]
        for r in rows
    )

    rank3_hits = sum(
        r["rank3_correct"]
        for r in rows
    )

    ranks = [
        int(r["target_rank"])
        for r in rows
    ]

    rr = [
        float(r["reciprocal_rank"])
        for r in rows
    ]

    margins = [
        float(
            r[
                "target_vs_hardest_distractor_margin"
            ]
        )
        for r in rows
    ]

    positive_margins = sum(
        x > 0
        for x in margins
    )

    improved = sum(
        int(r["m2_rank_change_vs_m1"]) > 0
        for r in rows
    )

    unchanged = sum(
        int(r["m2_rank_change_vs_m1"]) == 0
        for r in rows
    )

    worsened = sum(
        int(r["m2_rank_change_vs_m1"]) < 0
        for r in rows
    )

    active_rows = [
        r for r in rows
        if int(
            r["negative_memory_size"]
        ) > 0
    ]

    active_improved = sum(
        int(
            r["m2_rank_change_vs_m1"]
        ) > 0
        for r in active_rows
    )

    active_worsened = sum(
        int(
            r["m2_rank_change_vs_m1"]
        ) < 0
        for r in active_rows
    )

    print()
    print("-" * 118)

    print(
        f"M2 Rank-1 accuracy: "
        f"{rank1_hits}/{n} "
        f"= {rank1_hits/n:.3f}"
    )

    print(
        f"M2 Rank-3 accuracy: "
        f"{rank3_hits}/{n} "
        f"= {rank3_hits/n:.3f}"
    )

    print(
        f"M2 mean target rank: "
        f"{mean(ranks):.2f}"
    )

    print(
        f"M2 mean reciprocal rank: "
        f"{mean(rr):.3f}"
    )

    print(
        f"M2 positive target margin: "
        f"{positive_margins}/{n}"
    )

    print(
        f"M2 mean target margin: "
        f"{mean(margins):+.4f}"
    )

    print()
    print(
        f"Improved vs M1:  "
        f"{improved}/{n}"
    )

    print(
        f"Unchanged vs M1: "
        f"{unchanged}/{n}"
    )

    print(
        f"Worsened vs M1:  "
        f"{worsened}/{n}"
    )

    print()
    print(
        f"Events with active negative memory: "
        f"{len(active_rows)}/{n}"
    )

    print(
        f"Active-memory events improved: "
        f"{active_improved}/{len(active_rows)}"
    )

    print(
        f"Active-memory events worsened: "
        f"{active_worsened}/{len(active_rows)}"
    )

    print()
    print(
        "Reference results:"
    )

    print(
        "H0: Rank-1=5/10, Rank-3=6/10, "
        "mean rank=4.70, MRR=0.585"
    )

    print(
        "M1: Rank-1=5/10, Rank-3=5/10, "
        "mean rank=4.40, MRR=0.574"
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
