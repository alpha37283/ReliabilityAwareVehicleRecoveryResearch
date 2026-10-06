from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder
from recovery.recent_memory import load_recent_memory
from recovery.annotation_candidate_generator import get_frame_candidates
from recovery.real_candidate_generator import filter_real_candidates


ROOT = Path(__file__).resolve().parents[2]

EVENT_ID = "E004"

IMAGE_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/images"
)

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

OUTPUT_DIR = (
    ROOT
    / "results/phase1/phase1h1_e004_diagnostic"
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
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def draw_label(draw, xy, text, fill):

    x, y = xy

    bbox = draw.textbbox(
        (x, y),
        text,
    )

    pad = 3

    draw.rectangle(
        [
            bbox[0] - pad,
            bbox[1] - pad,
            bbox[2] + pad,
            bbox[3] + pad,
        ],
        fill="black",
    )

    draw.text(
        (x, y),
        text,
        fill=fill,
    )


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
        / EVENT_ID
        / "sam21"
        / "predictions.csv"
    )

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    # --------------------------------------------------
    # Same frozen M1 memory
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
    # Reappearance frame
    # --------------------------------------------------

    image_path = (
        IMAGE_ROOT
        / sequence
        / f"img{reappearance_frame:05d}.jpg"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------
    # Frozen Phase 1G candidate generation
    # --------------------------------------------------

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

    # --------------------------------------------------
    # M1 ranking
    # --------------------------------------------------

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
            "crop": crop,
            "detector_confidence":
                candidate["confidence"],
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

    gt_objects = get_frame_candidates(
        str(xml_path),
        reappearance_frame,
    )

    target_gt = next(
        obj
        for obj in gt_objects
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
            >= TARGET_IOU_THRESHOLD
        )

    # --------------------------------------------------
    # Print ranking
    # --------------------------------------------------

    print(
        "PHASE 1H-1 — E004 FAILURE VISUALIZATION"
    )
    print("=" * 100)

    print(
        f"Sequence: {sequence}"
    )

    print(
        f"Reappearance frame: {reappearance_frame}"
    )

    print(
        f"M1 memory size: {len(memory)}"
    )

    print(
        f"Candidate count: {len(scored)}"
    )

    print()

    for candidate in scored[:10]:

        print(
            f"Rank {candidate['rank']:2d} | "
            f"score={candidate['score']:.4f} | "
            f"GT IoU={candidate['target_iou']:.4f} | "
            f"is_target={candidate['is_target']} | "
            f"box={candidate['box_xyxy']}"
        )

    # --------------------------------------------------
    # Full-frame visualization
    # --------------------------------------------------

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    # GT target shown independently after ranking.
    gx1, gy1, gx2, gy2 = target_gt_box

    draw.rectangle(
        [gx1, gy1, gx2, gy2],
        outline="lime",
        width=4,
    )

    draw_label(
        draw,
        (gx1, max(2, gy1 - 18)),
        "GT TARGET",
        "lime",
    )

    # Draw Top-5 ranking.
    for candidate in scored[:5]:

        x1, y1, x2, y2 = (
            candidate["box_xyxy"]
        )

        # Rank-1 red, other ranked candidates yellow.
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
            f"{candidate['score']:.3f} "
            f"IoU={candidate['target_iou']:.2f}"
        )

        draw_label(
            draw,
            (x1, y1 + 2),
            label,
            color,
        )

    frame_output = (
        OUTPUT_DIR
        / "E004_ranked_candidates.jpg"
    )

    annotated.save(
        frame_output,
        quality=95,
    )

    # --------------------------------------------------
    # Top-5 candidate crop sheet
    # --------------------------------------------------

    top = scored[:5]

    cell_width = 260
    cell_height = 190
    title_height = 50

    sheet = Image.new(
        "RGB",
        (
            cell_width * len(top),
            cell_height + title_height,
        ),
        "white",
    )

    sheet_draw = ImageDraw.Draw(sheet)

    for index, candidate in enumerate(top):

        crop = candidate["crop"].copy()

        crop.thumbnail(
            (
                cell_width - 20,
                cell_height - 20,
            )
        )

        x_offset = (
            index * cell_width
            + (cell_width - crop.width) // 2
        )

        y_offset = (
            title_height
            + (cell_height - crop.height) // 2
        )

        sheet.paste(
            crop,
            (x_offset, y_offset),
        )

        label = (
            f"R{candidate['rank']} "
            f"score={candidate['score']:.3f} "
            f"GT={candidate['is_target']}"
        )

        sheet_draw.text(
            (
                index * cell_width + 8,
                10,
            ),
            label,
            fill="black",
        )

    sheet_output = (
        OUTPUT_DIR
        / "E004_top5_crops.jpg"
    )

    sheet.save(
        sheet_output,
        quality=95,
    )

    print()
    print("-" * 100)

    print(
        f"Saved full frame: "
        f"{frame_output}"
    )

    print(
        f"Saved crop sheet: "
        f"{sheet_output}"
    )


if __name__ == "__main__":
    main()
