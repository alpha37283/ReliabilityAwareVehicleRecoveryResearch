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

ELIGIBLE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h1_m2_failure_diagnostic.csv"
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


def main():

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    rows = []

    print(
        "PHASE 1H-1 — M2 FAILURE DIAGNOSTIC"
    )
    print("=" * 120)

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
        # M1 positive memory
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
            encoder.encode(entry["crop"])
            for entry in positive_memory
        ])

        # --------------------------------------------------
        # M2 negative memory
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

        # Diagnostic only for active M2 events.
        if not negative_memory:
            continue

        negative_embeddings = torch.stack([
            entry["embedding"]
            for entry in negative_memory
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
            imgsz=IMGSZ,
            conf=CONF,
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

            positive_score = float(
                torch.max(
                    torch.mv(
                        positive_embeddings,
                        embedding,
                    )
                ).item()
            )

            negative_score = float(
                torch.max(
                    torch.mv(
                        negative_embeddings,
                        embedding,
                    )
                ).item()
            )

            final_score = (
                positive_score
                - negative_score
            )

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

        # --------------------------------------------------
        # GT only AFTER scoring
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
            obj
            for obj in gt
            if int(obj["target_id"]) == target_id
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
            x for x in scored
            if x["is_target"]
        ]

        wrong = [
            x for x in scored
            if not x["is_target"]
        ]

        target = max(
            targets,
            key=lambda x:
                x["final_score"],
        )

        strongest_wrong = max(
            wrong,
            key=lambda x:
                x["final_score"],
        )

        # Positive means the true target receives a
        # LARGER negative-memory penalty than the wrong candidate.
        penalty_disadvantage = (
            target["negative_score"]
            - strongest_wrong["negative_score"]
        )

        # Compare positive evidence before negative penalty.
        positive_advantage = (
            target["positive_score"]
            - strongest_wrong["positive_score"]
        )

        final_advantage = (
            target["final_score"]
            - strongest_wrong["final_score"]
        )

        rows.append({
            "event_id":
                event_id,

            "negative_memory_size":
                len(negative_memory),

            "target_positive":
                round(
                    target["positive_score"],
                    6,
                ),

            "target_negative":
                round(
                    target["negative_score"],
                    6,
                ),

            "target_final":
                round(
                    target["final_score"],
                    6,
                ),

            "wrong_positive":
                round(
                    strongest_wrong[
                        "positive_score"
                    ],
                    6,
                ),

            "wrong_negative":
                round(
                    strongest_wrong[
                        "negative_score"
                    ],
                    6,
                ),

            "wrong_final":
                round(
                    strongest_wrong[
                        "final_score"
                    ],
                    6,
                ),

            "positive_advantage_target_minus_wrong":
                round(
                    positive_advantage,
                    6,
                ),

            "negative_penalty_disadvantage":
                round(
                    penalty_disadvantage,
                    6,
                ),

            "final_advantage_target_minus_wrong":
                round(
                    final_advantage,
                    6,
                ),

            "target_penalized_more":
                penalty_disadvantage > 0,

            "target_wins_after_penalty":
                final_advantage > 0,
        })

        print(
            f"{event_id} | "
            f"Tpos={target['positive_score']:.4f} "
            f"Tneg={target['negative_score']:.4f} "
            f"Tfinal={target['final_score']:+.4f} | "
            f"Wpos={strongest_wrong['positive_score']:.4f} "
            f"Wneg={strongest_wrong['negative_score']:.4f} "
            f"Wfinal={strongest_wrong['final_score']:+.4f} | "
            f"pos_adv={positive_advantage:+.4f} | "
            f"penalty_disadv={penalty_disadvantage:+.4f} | "
            f"final_adv={final_advantage:+.4f}"
        )

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

    print()
    print("-" * 120)

    penalized_more = sum(
        row["target_penalized_more"]
        for row in rows
    )

    target_wins = sum(
        row["target_wins_after_penalty"]
        for row in rows
    )

    positive_wins = sum(
        float(
            row[
                "positive_advantage_target_minus_wrong"
            ]
        ) > 0
        for row in rows
    )

    print(
        f"Active M2 events: "
        f"{len(rows)}"
    )

    print(
        f"Target had stronger positive evidence "
        f"than strongest wrong candidate: "
        f"{positive_wins}/{len(rows)}"
    )

    print(
        f"Target received larger negative-memory "
        f"penalty than strongest wrong candidate: "
        f"{penalized_more}/{len(rows)}"
    )

    print(
        f"Target still beats strongest wrong "
        f"candidate after subtraction: "
        f"{target_wins}/{len(rows)}"
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
