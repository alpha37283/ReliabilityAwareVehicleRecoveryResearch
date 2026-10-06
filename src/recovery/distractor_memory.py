import csv
import math
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from recovery.real_candidate_generator import filter_real_candidates


STABLE_IOU = 0.63
STABLE_CENTER_JUMP = 0.12

SIMILARITY_THRESHOLD = 0.80
DISTANCE_THRESHOLD = 3.0
TARGET_OVERLAP_THRESHOLD = 0.30

FRAME_STEP = 5
NEGATIVE_CAPACITY = 3


def parse_bool(value):
    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def box_iou(a, b):

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

    return inter / union if union > 0 else 0.0


def center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def diagonal(box):

    x1, y1, x2, y2 = box

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    return math.sqrt(w*w + h*h)


def normalized_jump(previous_box, current_box):

    px, py = center(previous_box)
    cx, cy = center(current_box)

    distance = math.sqrt(
        (cx-px)**2
        + (cy-py)**2
    )

    return (
        distance
        / diagonal(previous_box)
    )


def normalized_distance(target_box, other_box):

    tx, ty = center(target_box)
    ox, oy = center(other_box)

    distance = math.sqrt(
        (tx-ox)**2
        + (ty-oy)**2
    )

    return (
        distance
        / diagonal(target_box)
    )


def clamp_box(box, width, height):

    x1, y1, x2, y2 = box

    return [
        int(max(0, min(width, round(float(x1))))),
        int(max(0, min(height, round(float(y1))))),
        int(max(0, min(width, round(float(x2))))),
        int(max(0, min(height, round(float(y2))))),
    ]


def crop_box(image, box):

    box = clamp_box(
        box,
        image.width,
        image.height,
    )

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return None, None

    return image.crop(box), box


def load_tracker_predictions(predictions_csv):

    predictions = {}

    with Path(predictions_csv).open(newline="") as f:

        for row in csv.DictReader(f):

            if not parse_bool(row["present"]):
                continue

            frame = int(row["frame_number"])

            try:
                box = [
                    float(row["box_x1"]),
                    float(row["box_y1"]),
                    float(row["box_x2"]),
                    float(row["box_y2"]),
                ]
            except Exception:
                continue

            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            predictions[frame] = box

    return predictions


def build_distractor_memory(
    predictions_csv,
    image_dir,
    selection_frame,
    memory_cutoff_frame,
    encoder,
    detector=None,
    capacity=NEGATIVE_CAPACITY,
    frame_step=FRAME_STEP,
):

    """
    Build M2 negative distractor memory using runtime-available
    information only.

    No ground-truth identity information is used.

    Returns a list of dictionaries sorted from hardest to easier
    distractor observation.
    """

    image_dir = Path(image_dir)

    predictions = load_tracker_predictions(
        predictions_csv
    )

    if detector is None:
        detector = YOLO("yolo11n.pt")

    sampled_frames = list(
        range(
            selection_frame,
            memory_cutoff_frame,
            frame_step,
        )
    )

    final_frame = memory_cutoff_frame - 1

    if (
        final_frame >= selection_frame
        and final_frame not in sampled_frames
    ):
        sampled_frames.append(final_frame)

    sampled_frames = sorted(
        set(sampled_frames)
    )

    encounters = []

    for frame in sampled_frames:

        target_box = predictions.get(frame)

        if target_box is None:
            continue

        # ----------------------------------------------
        # Reliability gate
        # ----------------------------------------------

        if frame == selection_frame:
            reliable = True

        else:

            previous_frames = [
                f for f in predictions
                if f < frame
            ]

            if not previous_frames:
                continue

            previous_frame = max(
                previous_frames
            )

            previous_box = predictions[
                previous_frame
            ]

            temporal_iou = box_iou(
                previous_box,
                target_box,
            )

            jump = normalized_jump(
                previous_box,
                target_box,
            )

            reliable = (
                temporal_iou >= STABLE_IOU
                and
                jump <= STABLE_CENTER_JUMP
            )

        if not reliable:
            continue

        image_path = (
            image_dir
            / f"img{frame:05d}.jpg"
        )

        if not image_path.exists():
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        target_crop, target_box_int = crop_box(
            image,
            target_box,
        )

        if target_crop is None:
            continue

        target_embedding = encoder.encode(
            target_crop
        )

        # ----------------------------------------------
        # Vehicle detections
        # ----------------------------------------------

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

            class_id = int(box.cls[0])

            raw.append({
                "box_xyxy":
                    box.xyxy[0].tolist(),

                "confidence":
                    float(box.conf[0]),

                "class_id":
                    class_id,

                "class_name":
                    result.names[class_id],
            })

        detections = filter_real_candidates(
            raw,
            target_box,
        )

        frame_encounters = []

        for detection in detections:

            distractor_box = (
                detection["box_xyxy"]
            )

            # Ignore detector boxes corresponding
            # to the tracked target.
            if (
                box_iou(
                    target_box,
                    distractor_box,
                )
                >= TARGET_OVERLAP_THRESHOLD
            ):
                continue

            (
                distractor_crop,
                distractor_box_int,
            ) = crop_box(
                image,
                distractor_box,
            )

            if distractor_crop is None:
                continue

            distractor_embedding = (
                encoder.encode(
                    distractor_crop
                )
            )

            similarity = float(
                torch.dot(
                    target_embedding,
                    distractor_embedding,
                ).item()
            )

            distance = normalized_distance(
                target_box,
                distractor_box,
            )

            if (
                similarity
                < SIMILARITY_THRESHOLD
            ):
                continue

            if distance > DISTANCE_THRESHOLD:
                continue

            frame_encounters.append({
                "frame_number":
                    frame,

                "target_box":
                    target_box_int,

                "distractor_box":
                    distractor_box_int,

                "target_distractor_similarity":
                    similarity,

                "normalized_distance":
                    distance,

                "detector_confidence":
                    detection["confidence"],

                "crop":
                    distractor_crop,

                "embedding":
                    distractor_embedding,
            })

        # Keep only the hardest distractor
        # observation from this sampled frame.
        if frame_encounters:

            hardest = max(
                frame_encounters,
                key=lambda x:
                    x[
                        "target_distractor_similarity"
                    ],
            )

            encounters.append(hardest)

    # Hardest observations first.
    encounters.sort(
        key=lambda x:
            x["target_distractor_similarity"],
        reverse=True,
    )

    return encounters[:capacity]
