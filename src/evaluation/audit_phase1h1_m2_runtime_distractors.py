import csv
import math
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder
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

OUTPUT = (
    ROOT
    / "results/phase1/phase1h1_m2_runtime_distractor_audit.csv"
)

IMGSZ = 640
CONF = 0.20

FRAME_STEP = 5

# Phase 1F L2 stable-geometry criteria
STABLE_IOU = 0.63
STABLE_CENTER_JUMP = 0.12

# Provisional M2 opportunity thresholds only.
SIMILARITY_THRESHOLD = 0.80
DISTANCE_THRESHOLD = 3.0

# Any YOLO detection overlapping the tracked target this much
# is treated as target-associated, not as a distractor.
TARGET_OVERLAP_THRESHOLD = 0.30


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

    area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)

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

    w = max(1.0, x2-x1)
    h = max(1.0, y2-y1)

    return math.sqrt(w*w + h*h)


def normalized_jump(previous_box, current_box):

    px, py = center(previous_box)
    cx, cy = center(current_box)

    distance = math.sqrt(
        (cx-px)**2 + (cy-py)**2
    )

    return distance / diagonal(previous_box)


def normalized_distance(target_box, other_box):

    tx, ty = center(target_box)
    ox, oy = center(other_box)

    distance = math.sqrt(
        (tx-ox)**2 + (ty-oy)**2
    )

    return distance / diagonal(target_box)


def clamp_box(box, width, height):

    x1, y1, x2, y2 = box

    return [
        int(max(0, min(width, round(float(x1))))),
        int(max(0, min(height, round(float(y1))))),
        int(max(0, min(width, round(float(x2))))),
        int(max(0, min(height, round(float(y2))))),
    ]


def crop(image, box):

    box = clamp_box(
        box,
        image.width,
        image.height,
    )

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop(box)


def load_predictions(event_id):

    path = (
        ROOT
        / "results/phase1"
        / event_id
        / "sam21"
        / "predictions.csv"
    )

    lookup = {}

    with path.open(newline="") as f:

        for row in csv.DictReader(f):

            frame = int(row["frame_number"])

            if not parse_bool(row["present"]):
                continue

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

            lookup[frame] = box

    return lookup


def main():

    detector = YOLO("yolo11n.pt")
    encoder = AppearanceEncoder()

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    output_rows = []

    print(
        "PHASE 1H-1 — M2 RUNTIME DISTRACTOR AUDIT"
    )
    print("=" * 110)

    for item in events:

        event_id = item["event_id"]
        event = load_event(event_id)

        sequence = event["sequence"]

        selection = int(
            event["selection_frame"]
        )

        occlusion = int(
            event["occlusion_start"]
        )

        predictions = load_predictions(
            event_id
        )

        sampled = list(
            range(
                selection,
                occlusion,
                FRAME_STEP,
            )
        )

        final_frame = occlusion - 1

        if (
            final_frame >= selection
            and final_frame not in sampled
        ):
            sampled.append(final_frame)

        sampled = sorted(set(sampled))

        reliable_frames = 0
        encounter_frames = 0

        best_similarity = None
        best_frame = None
        best_distance = None

        for frame in sampled:

            target_box = predictions.get(frame)

            if target_box is None:
                continue

            # ----------------------------------------------
            # Runtime reliability gate
            # ----------------------------------------------

            if frame == selection:
                reliable = True

            else:
                previous_frames = [
                    f for f in predictions
                    if f < frame
                ]

                if not previous_frames:
                    reliable = False
                else:
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

            reliable_frames += 1

            image_path = (
                IMAGE_ROOT
                / sequence
                / f"img{frame:05d}.jpg"
            )

            image = Image.open(
                image_path
            ).convert("RGB")

            target_crop = crop(
                image,
                target_box,
            )

            if target_crop is None:
                continue

            target_embedding = encoder.encode(
                target_crop
            )

            # ----------------------------------------------
            # Runtime vehicle detector
            # ----------------------------------------------

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

            # Same broad geometry plausibility logic as Phase 1G,
            # now relative to the current tracked target box.
            detections = filter_real_candidates(
                raw,
                target_box,
            )

            frame_best = None

            for detection in detections:

                det_box = detection[
                    "box_xyxy"
                ]

                # Exclude detections corresponding to
                # the currently tracked target.
                if (
                    box_iou(
                        target_box,
                        det_box,
                    )
                    >= TARGET_OVERLAP_THRESHOLD
                ):
                    continue

                other_crop = crop(
                    image,
                    det_box,
                )

                if other_crop is None:
                    continue

                other_embedding = (
                    encoder.encode(
                        other_crop
                    )
                )

                similarity = float(
                    torch.dot(
                        target_embedding,
                        other_embedding,
                    ).item()
                )

                distance = normalized_distance(
                    target_box,
                    det_box,
                )

                hard = (
                    similarity
                    >= SIMILARITY_THRESHOLD
                    and
                    distance
                    <= DISTANCE_THRESHOLD
                )

                candidate = {
                    "event_id":
                        event_id,

                    "frame_number":
                        frame,

                    "similarity":
                        similarity,

                    "normalized_distance":
                        distance,

                    "detector_confidence":
                        detection["confidence"],

                    "hard_distractor":
                        hard,
                }

                if (
                    frame_best is None
                    or similarity
                    > frame_best["similarity"]
                ):
                    frame_best = candidate

            if frame_best is None:
                continue

            output_rows.append(
                frame_best
            )

            if frame_best[
                "hard_distractor"
            ]:
                encounter_frames += 1

            if (
                best_similarity is None
                or
                frame_best["similarity"]
                > best_similarity
            ):
                best_similarity = (
                    frame_best["similarity"]
                )

                best_frame = frame

                best_distance = (
                    frame_best[
                        "normalized_distance"
                    ]
                )

        print(
            f"{event_id} | "
            f"sampled={len(sampled):2d} | "
            f"reliable={reliable_frames:2d} | "
            f"hard_encounters={encounter_frames:2d} | "
            f"best_sim="
            f"{best_similarity if best_similarity is not None else 0:.4f} | "
            f"best_frame={best_frame} | "
            f"distance="
            f"{best_distance if best_distance is not None else 0:.2f}"
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
            fieldnames=[
                "event_id",
                "frame_number",
                "similarity",
                "normalized_distance",
                "detector_confidence",
                "hard_distractor",
            ],
        )

        writer.writeheader()

        for row in output_rows:

            out = row.copy()

            out["similarity"] = round(
                out["similarity"],
                6,
            )

            out["normalized_distance"] = round(
                out["normalized_distance"],
                4,
            )

            out["detector_confidence"] = round(
                out["detector_confidence"],
                4,
            )

            writer.writerow(out)

    events_with_runtime_hard = {
        row["event_id"]
        for row in output_rows
        if row["hard_distractor"]
    }

    print()
    print("-" * 110)

    print(
        f"Events with deployable hard-distractor "
        f"encounter: "
        f"{len(events_with_runtime_hard)}/{len(events)}"
    )

    print(
        "No ground-truth identity was used to generate "
        "these runtime distractor encounters."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
