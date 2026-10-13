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
    "E008",
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
    / "results/phase1/phase1i_f7_review"
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


def draw_label(draw, x, y, text, color):

    bbox = draw.textbbox(
        (x, y),
        text,
    )

    draw.rectangle(
        [
            bbox[0] - 3,
            bbox[1] - 3,
            bbox[2] + 3,
            bbox[3] + 3,
        ],
        fill="black",
    )

    draw.text(
        (x, y),
        text,
        fill=color,
    )


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    print(
        "PHASE 1I — F7 AMBIGUITY MANUAL-REVIEW VISUALIZATION"
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

            embedding = encoder.encode(
                crop
            )

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
                "crop": crop,
                "confidence":
                    float(candidate["confidence"]),
            })

        scored.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        # --------------------------------------------------
        # GT after ranking
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

        target_rank = next(
            candidate["rank"]
            for candidate in scored
            if candidate["is_target"]
        )

        margin = (
            scored[0]["score"]
            - scored[1]["score"]
        )

        # --------------------------------------------------
        # Full-frame visualization
        # --------------------------------------------------

        annotated = image.copy()
        draw = ImageDraw.Draw(
            annotated
        )

        gx1, gy1, gx2, gy2 = target_gt_box

        draw.rectangle(
            [gx1, gy1, gx2, gy2],
            outline="lime",
            width=4,
        )

        draw_label(
            draw,
            gx1,
            max(2, gy1 - 18),
            "GT TARGET",
            "lime",
        )

        for candidate in scored[:5]:

            x1, y1, x2, y2 = (
                candidate["box_xyxy"]
            )

            color = (
                "red"
                if candidate["rank"] == 1
                else "yellow"
            )

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color,
                width=3,
            )

            label = (
                f"R{candidate['rank']} "
                f"S={candidate['score']:.3f}"
            )

            draw_label(
                draw,
                x1,
                y1 + 2,
                label,
                color,
            )

        frame_output = (
            OUTPUT_ROOT
            / f"{event_id}_ranked_frame.jpg"
        )

        annotated.save(
            frame_output,
            quality=95,
        )

        # --------------------------------------------------
        # Top-5 crop sheet
        # --------------------------------------------------

        top = scored[:5]

        cell_w = 260
        cell_h = 190
        title_h = 55

        sheet = Image.new(
            "RGB",
            (
                cell_w * len(top),
                cell_h + title_h,
            ),
            "white",
        )

        sheet_draw = ImageDraw.Draw(
            sheet
        )

        for i, candidate in enumerate(top):

            crop = candidate["crop"].copy()

            crop.thumbnail(
                (
                    cell_w - 20,
                    cell_h - 20,
                )
            )

            x = (
                i * cell_w
                + (cell_w - crop.width) // 2
            )

            y = (
                title_h
                + (cell_h - crop.height) // 2
            )

            sheet.paste(
                crop,
                (x, y),
            )

            text = (
                f"R{candidate['rank']} "
                f"S={candidate['score']:.3f} "
                f"GT={candidate['is_target']}"
            )

            sheet_draw.text(
                (
                    i * cell_w + 8,
                    10,
                ),
                text,
                fill="black",
            )

        crop_output = (
            OUTPUT_ROOT
            / f"{event_id}_top5_crops.jpg"
        )

        sheet.save(
            crop_output,
            quality=95,
        )

        print(
            f"{event_id} | "
            f"frame={reappearance} | "
            f"target_rank={target_rank} | "
            f"top1={scored[0]['score']:.4f} | "
            f"top2={scored[1]['score']:.4f} | "
            f"margin={margin:.4f} | "
            f"candidates={len(scored)}"
        )

        print(
            f"  frame: {frame_output}"
        )

        print(
            f"  crops: {crop_output}"
        )

    print()
    print("-" * 110)

    print(
        "Manual F7 question:"
    )

    print(
        "Do multiple top-ranked candidates genuinely look "
        "plausible as the selected vehicle, rather than the "
        "failure being a clearly wrong appearance match?"
    )

    print()
    print(
        "No F7 labels are assigned by this script."
    )


if __name__ == "__main__":
    main()
