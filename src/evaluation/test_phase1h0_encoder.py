from pathlib import Path

import torch
from PIL import Image, ImageDraw

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder


ROOT = Path(__file__).resolve().parents[2]

EVENT_ID = "E011"

IMAGE_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/images"
)

OUTPUT_DIR = (
    ROOT
    / "results/phase1/phase1h0_debug"
)


def get_xyxy(selection_box):

    if isinstance(selection_box, dict):
        return selection_box["xyxy"]

    return selection_box


def clamp_box(box, width, height):

    x1, y1, x2, y2 = box

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    return [
        int(round(x1)),
        int(round(y1)),
        int(round(x2)),
        int(round(y2)),
    ]


def main():

    event = load_event(EVENT_ID)

    sequence = event["sequence"]
    frame = int(event["selection_frame"])

    selection_box = get_xyxy(
        event["selection_box"]
    )

    image_path = (
        IMAGE_ROOT
        / sequence
        / f"img{frame:05d}.jpg"
    )

    image = Image.open(image_path).convert("RGB")

    box = clamp_box(
        selection_box,
        image.width,
        image.height,
    )

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid selection box: {box}"
        )

    target_crop = image.crop(box)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop_path = (
        OUTPUT_DIR
        / f"{EVENT_ID}_reference_crop.jpg"
    )

    target_crop.save(crop_path)

    # Also save original frame with selection box
    # for visual verification.
    debug_frame = image.copy()

    draw = ImageDraw.Draw(debug_frame)

    draw.rectangle(
        box,
        outline="red",
        width=3,
    )

    boxed_path = (
        OUTPUT_DIR
        / f"{EVENT_ID}_selection_debug.jpg"
    )

    debug_frame.save(boxed_path)

    encoder = AppearanceEncoder()

    embedding = encoder.encode(
        target_crop
    )

    norm = torch.linalg.vector_norm(
        embedding
    ).item()

    print("PHASE 1H-0 — APPEARANCE ENCODER TEST")
    print("=" * 65)

    print(f"Event:              {EVENT_ID}")
    print(f"Sequence:           {sequence}")
    print(f"Selection frame:    {frame}")
    print(f"Selection box:      {box}")

    print(
        f"Crop size:          "
        f"{target_crop.width} x "
        f"{target_crop.height}"
    )

    print(
        f"Encoder device:     "
        f"{encoder.device}"
    )

    print(
        f"Embedding shape:    "
        f"{tuple(embedding.shape)}"
    )

    print(
        f"Embedding dimension:"
        f" {embedding.numel()}"
    )

    print(
        f"L2 norm:            "
        f"{norm:.6f}"
    )

    print()
    print(f"Reference crop:     {crop_path}")
    print(f"Selection debug:    {boxed_path}")

    assert embedding.numel() == 2048

    assert abs(norm - 1.0) < 1e-5

    print()
    print("Encoder verification: PASS")


if __name__ == "__main__":
    main()
