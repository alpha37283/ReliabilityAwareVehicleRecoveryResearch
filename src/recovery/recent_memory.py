import csv
import math
from pathlib import Path

from PIL import Image


def parse_bool(value):
    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def valid_box(box):

    if box is None:
        return False

    x1, y1, x2, y2 = box

    if not all(
        math.isfinite(float(v))
        for v in box
    ):
        return False

    return (
        float(x2) > float(x1)
        and float(y2) > float(y1)
    )


def clamp_box(box, width, height):

    x1, y1, x2, y2 = box

    return [
        int(max(0, min(width, round(float(x1))))),
        int(max(0, min(height, round(float(y1))))),
        int(max(0, min(width, round(float(x2))))),
        int(max(0, min(height, round(float(y2))))),
    ]


def load_recent_memory(
    predictions_csv,
    image_dir,
    selection_frame,
    memory_cutoff_frame,
    selection_box,
    capacity=5,
):

    """
    M1 ordinary recent FIFO appearance memory.

    Memory contains:
        1. Original user-selection crop.
        2. Up to capacity-1 most recent valid tracker crops
           strictly before the reappearance frame.

    No reliability gating.
    No GT identity checks.
    No distractor awareness.
    """

    predictions_csv = Path(predictions_csv)
    image_dir = Path(image_dir)

    # --------------------------------------------------
    # Original selection crop
    # --------------------------------------------------

    selection_path = (
        image_dir
        / f"img{selection_frame:05d}.jpg"
    )

    selection_image = Image.open(
        selection_path
    ).convert("RGB")

    selection_box = clamp_box(
        selection_box,
        selection_image.width,
        selection_image.height,
    )

    memory = [{
        "frame_number": selection_frame,
        "source": "selection",
        "box_xyxy": selection_box,
        "crop": selection_image.crop(
            selection_box
        ),
    }]

    # --------------------------------------------------
    # Gather ordinary recent tracker predictions
    # --------------------------------------------------

    valid_rows = []

    with predictions_csv.open(newline="") as f:

        for row in csv.DictReader(f):

            frame = int(row["frame_number"])

            if frame < selection_frame:
                continue

            if frame >= memory_cutoff_frame:
                continue

            if not parse_bool(row["present"]):
                continue

            box = [
                float(row["box_x1"]),
                float(row["box_y1"]),
                float(row["box_x2"]),
                float(row["box_y2"]),
            ]

            if not valid_box(box):
                continue

            valid_rows.append(
                (frame, box)
            )

    # FIFO recent memory:
    # retain only latest capacity-1 tracker observations.
    valid_rows = valid_rows[
        -(capacity - 1):
    ]

    for frame, box in valid_rows:

        image_path = (
            image_dir
            / f"img{frame:05d}.jpg"
        )

        if not image_path.exists():
            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        box_int = clamp_box(
            box,
            image.width,
            image.height,
        )

        x1, y1, x2, y2 = box_int

        if x2 <= x1 or y2 <= y1:
            continue

        memory.append({
            "frame_number": frame,
            "source": "recent_tracker",
            "box_xyxy": box_int,
            "crop": image.crop(
                box_int
            ),
        })

    return memory
