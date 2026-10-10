import csv
import math
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
    / "results/phase1/phase1h3_observation_quality.csv"
)

TARGET_IOU = 0.30


def get_xyxy(box):
    return box["xyxy"] if isinstance(box, dict) else box


def area(box):
    x1, y1, x2, y2 = box
    return max(1.0, x2-x1) * max(1.0, y2-y1)


def aspect_ratio(box):
    x1, y1, x2, y2 = box
    w = max(1.0, x2-x1)
    h = max(1.0, y2-y1)
    return w / h


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


def clamp_box(box, w, h):

    x1, y1, x2, y2 = box

    return [
        int(max(0, min(w, round(float(x1))))),
        int(max(0, min(h, round(float(y1))))),
        int(max(0, min(w, round(float(x2))))),
        int(max(0, min(h, round(float(y2))))),
    ]


def quality_metrics(
    candidate,
    embedding,
    memory_embeddings,
    selection_box,
):

    similarities = torch.mv(
        memory_embeddings,
        embedding,
    )

    max_sim = float(
        torch.max(similarities).item()
    )

    mean_sim = float(
        torch.mean(similarities).item()
    )

    std_sim = (
        float(
            torch.std(
                similarities,
                unbiased=False,
            ).item()
        )
        if len(similarities) > 1
        else 0.0
    )

    box = candidate["box_xyxy"]

    area_ratio = (
        area(box)
        / area(selection_box)
    )

    ar_ratio = (
        aspect_ratio(box)
        / aspect_ratio(selection_box)
    )

    abs_log_area_change = abs(
        math.log(max(area_ratio, 1e-6))
    )

    abs_log_ar_change = abs(
        math.log(max(ar_ratio, 1e-6))
    )

    return {
        "max_similarity":
            max_sim,

        "mean_similarity":
            mean_sim,

        "similarity_std":
            std_sim,

        "detector_confidence":
            float(candidate["confidence"]),

        "area_ratio":
            area_ratio,

        "abs_log_area_change":
            abs_log_area_change,

        "aspect_ratio_ratio":
            ar_ratio,

        "abs_log_aspect_change":
            abs_log_ar_change,
    }


def main():

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    rows = []

    print(
        "PHASE 1H-3 — OBSERVATION QUALITY DIAGNOSTIC"
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
        # Frozen M1 memory
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

            metrics = quality_metrics(
                candidate={
                    **candidate,
                    "box_xyxy": box,
                },
                embedding=embedding,
                memory_embeddings=memory_embeddings,
                selection_box=selection_box,
            )

            scored.append({
                "box_xyxy": box,
                **metrics,
            })

        # M1 ranking = maximum positive-memory similarity.
        scored.sort(
            key=lambda x:
                x["max_similarity"],
            reverse=True,
        )

        top1 = scored[0]

        # --------------------------------------------------
        # GT only AFTER ranking
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
                >= TARGET_IOU
            )

        target_candidates = [
            x for x in scored
            if x["is_target"]
        ]

        target = max(
            target_candidates,
            key=lambda x:
                x["max_similarity"],
        )

        top1_correct = (
            top1["target_iou"]
            >= TARGET_IOU
        )

        target_rank = next(
            index
            for index, x in enumerate(
                scored,
                start=1,
            )
            if x is target
        )

        row = {
            "event_id":
                event_id,

            "top1_correct":
                top1_correct,

            "target_rank":
                target_rank,

            "memory_size":
                len(memory),

            # ---------------- Top1 ----------------

            "top1_max_similarity":
                round(
                    top1["max_similarity"],
                    6,
                ),

            "top1_mean_similarity":
                round(
                    top1["mean_similarity"],
                    6,
                ),

            "top1_similarity_std":
                round(
                    top1["similarity_std"],
                    6,
                ),

            "top1_detector_confidence":
                round(
                    top1["detector_confidence"],
                    6,
                ),

            "top1_area_ratio":
                round(
                    top1["area_ratio"],
                    6,
                ),

            "top1_abs_log_area_change":
                round(
                    top1["abs_log_area_change"],
                    6,
                ),

            "top1_aspect_ratio_ratio":
                round(
                    top1["aspect_ratio_ratio"],
                    6,
                ),

            "top1_abs_log_aspect_change":
                round(
                    top1[
                        "abs_log_aspect_change"
                    ],
                    6,
                ),

            # ---------------- True target ----------------

            "target_max_similarity":
                round(
                    target["max_similarity"],
                    6,
                ),

            "target_mean_similarity":
                round(
                    target["mean_similarity"],
                    6,
                ),

            "target_similarity_std":
                round(
                    target["similarity_std"],
                    6,
                ),

            "target_detector_confidence":
                round(
                    target["detector_confidence"],
                    6,
                ),

            "target_area_ratio":
                round(
                    target["area_ratio"],
                    6,
                ),

            "target_abs_log_area_change":
                round(
                    target[
                        "abs_log_area_change"
                    ],
                    6,
                ),

            "target_aspect_ratio_ratio":
                round(
                    target[
                        "aspect_ratio_ratio"
                    ],
                    6,
                ),

            "target_abs_log_aspect_change":
                round(
                    target[
                        "abs_log_aspect_change"
                    ],
                    6,
                ),
        }

        rows.append(row)

        print(
            f"{event_id} | "
            f"correct={str(top1_correct):5} | "
            f"rank={target_rank:2d} | "
            f"Tsim={target['max_similarity']:.4f} | "
            f"Tconf={target['detector_confidence']:.3f} | "
            f"Tarea={target['area_ratio']:.2f}x | "
            f"Tar={target['aspect_ratio_ratio']:.2f}x | "
            f"Tstd={target['similarity_std']:.4f} | "
            f"TOPsim={top1['max_similarity']:.4f}"
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
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # Summary correct vs wrong
    # --------------------------------------------------

    correct = [
        r for r in rows
        if r["top1_correct"]
    ]

    wrong = [
        r for r in rows
        if not r["top1_correct"]
    ]

    print()
    print("-" * 125)

    print(
        f"Correct Rank-1: "
        f"{len(correct)}/{len(rows)}"
    )

    print(
        f"Wrong Rank-1: "
        f"{len(wrong)}/{len(rows)}"
    )

    metrics = [
        (
            "target_max_similarity",
            "target max similarity",
        ),
        (
            "target_detector_confidence",
            "target detector confidence",
        ),
        (
            "target_abs_log_area_change",
            "target |log area change|",
        ),
        (
            "target_abs_log_aspect_change",
            "target |log aspect change|",
        ),
        (
            "target_similarity_std",
            "target memory similarity std",
        ),
    ]

    print()

    for key, label in metrics:

        correct_values = [
            float(r[key])
            for r in correct
        ]

        wrong_values = [
            float(r[key])
            for r in wrong
        ]

        print(
            f"{label}:"
        )

        print(
            f"  correct mean = "
            f"{mean(correct_values):.4f}"
        )

        print(
            f"  wrong mean   = "
            f"{mean(wrong_values):.4f}"
        )

    print()
    print(
        "No quality threshold or routing rule "
        "is selected in this diagnostic."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
