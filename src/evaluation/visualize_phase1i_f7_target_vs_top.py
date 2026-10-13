from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder
from recovery.recent_memory import load_recent_memory
from recovery.annotation_candidate_generator import get_frame_candidates
from recovery.real_candidate_generator import filter_real_candidates


ROOT = Path(__file__).resolve().parents[2]

EVENT_IDS = [
    "E001",
    "E003",
    "E011",
]

IMAGE_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/images"
)

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

OUTPUT_ROOT = (
    ROOT
    / "results/phase1/phase1i_f7_target_vs_top"
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

    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def make_cell(crop, title, width=300, height=220):

    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    crop = crop.copy()

    crop.thumbnail(
        (width - 20, height - 55)
    )

    x = (width - crop.width) // 2
    y = 45 + (height - 45 - crop.height) // 2

    canvas.paste(
        crop,
        (x, y),
    )

    draw.text(
        (8, 10),
        title,
        fill="black",
    )

    return canvas


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    print(
        "PHASE 1I — F7 TRUE TARGET VS TOP WRONG CANDIDATES"
    )
    print("=" * 110)

    for event_id in EVENT_IDS:

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

            emb = encoder.encode(crop)

            score = float(
                torch.max(
                    torch.mv(
                        memory_embeddings,
                        emb,
                    )
                ).item()
            )

            scored.append({
                "box_xyxy": box,
                "score": score,
                "crop": crop,
            })

        scored.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        # --------------------------------------------------
        # GT only after ranking
        # --------------------------------------------------

        xml_path = (
            XML_ROOT
            / f"{sequence}.xml"
        )

        objects = get_frame_candidates(
            str(xml_path),
            reappearance,
        )

        target_gt = next(
            obj
            for obj in objects
            if int(obj["target_id"]) == target_id
        )

        target_gt_box = clamp_box(
            target_gt["box_xyxy"],
            image.width,
            image.height,
        )

        for rank, candidate in enumerate(
            scored,
            start=1,
        ):

            candidate["rank"] = rank

            candidate["target_iou"] = iou(
                candidate["box_xyxy"],
                target_gt_box,
            )

            candidate["is_target"] = (
                candidate["target_iou"]
                >= TARGET_IOU
            )

        target_candidate = max(
            [
                x for x in scored
                if x["is_target"]
            ],
            key=lambda x: x["score"],
        )

        wrong_candidates = [
            x for x in scored
            if not x["is_target"]
        ][:3]

        # --------------------------------------------------
        # Build comparison sheet
        # --------------------------------------------------

        cells = []

        cells.append(
            make_cell(
                target_candidate["crop"],
                (
                    f"TRUE TARGET | "
                    f"rank={target_candidate['rank']} | "
                    f"score={target_candidate['score']:.3f}"
                ),
            )
        )

        for wrong in wrong_candidates:

            cells.append(
                make_cell(
                    wrong["crop"],
                    (
                        f"WRONG R{wrong['rank']} | "
                        f"score={wrong['score']:.3f}"
                    ),
                )
            )

        sheet = Image.new(
            "RGB",
            (
                300 * len(cells),
                220,
            ),
            "white",
        )

        for i, cell in enumerate(cells):

            sheet.paste(
                cell,
                (i * 300, 0),
            )

        output = (
            OUTPUT_ROOT
            / f"{event_id}_target_vs_top3.jpg"
        )

        sheet.save(
            output,
            quality=95,
        )

        print(
            f"{event_id} | "
            f"target_rank={target_candidate['rank']} | "
            f"target_score={target_candidate['score']:.4f} | "
            f"top_wrong_scores="
            + ", ".join(
                f"{x['score']:.4f}"
                for x in wrong_candidates
            )
        )

        print(
            f"  saved: {output}"
        )

    print()
    print("-" * 110)

    print(
        "Manual F7 decision:"
    )

    print(
        "YES only if the true target itself remains genuinely "
        "confusable with the highest-ranked wrong candidates."
    )

    print(
        "If the wrong candidates are visibly different from the "
        "true target, classify primarily as F5 rather than F7."
    )


if __name__ == "__main__":
    main()
