
from pathlib import Path
import csv
import math
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data/raw/UA-DETRAC"
CSV_DIR = ROOT / "data/processed/pilot_events"
OUTPUT_DIR = ROOT / "results/visualizations"

# # Initial inspection: two events from each sequence.
# EVENTS = {
#     "MVI_39311": [5, 37],
#     "MVI_39401": [92, 75],
#     "MVI_40761": [2, 4],
# }

# # Expected intervals distinguish repeated events for the same target.
# EXPECTED_INTERVALS = {
#     ("MVI_39311", 5): (208, 258),
#     ("MVI_39311", 37): (1144, 1224),
#     ("MVI_39401", 92): (896, 1016),
#     ("MVI_39401", 75): (634, 718),
#     ("MVI_40761", 2): (1509, 1551),
#     ("MVI_40761", 4): (632, 722),
# }



EVENTS = {
    "MVI_39311": [35],
    "MVI_39401": [39],
    "MVI_40761": [24],
}

EXPECTED_INTERVALS = {
    ("MVI_39311", 35): (946, 1093),
    ("MVI_39401", 39): (243, 347),
    ("MVI_40761", 24): (310, 370),
}

PANEL_WIDTH = 600
PANEL_HEIGHT = 520

FULL_FRAME_SIZE = (580, 326)
CROP_SIZE = (580, 300)

# The crop includes the target and surrounding traffic.
CROP_PADDING = 1.5


def locate_xml(sequence):
    for folder in [
        "test_xml/DETRAC-Test-Annotations-XML",
        "train_xml/DETRAC-Train-Annotations-XML",
    ]:
        path = (
            DATASET / "annotations" / folder
            / f"{sequence}.xml"
        )

        if path.exists():
            return path

    raise FileNotFoundError(
        f"No XML annotation found for {sequence}"
    )


def load_annotations(sequence):
    xml_path = locate_xml(sequence)

    root = ET.parse(xml_path).getroot()

    annotations = {}

    for frame in root.findall("frame"):
        frame_num = int(frame.get("num"))

        annotations[frame_num] = {
            int(target.get("id")): target
            for target in frame.findall(
                "./target_list/target"
            )
        }

    return annotations


def load_image_paths(sequence):
    image_dir = DATASET / "images" / sequence

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    result = {}

    for path in image_dir.iterdir():
        if path.suffix.lower() not in {
            ".jpg", ".jpeg", ".png"
        }:
            continue

        digits = "".join(
            character for character in path.stem
            if character.isdigit()
        )

        if digits:
            result[int(digits)] = path

    return result


def get_box(target):
    if target is None:
        return None

    element = target.find("box")

    if element is None:
        return None

    x = float(element.get("left"))
    y = float(element.get("top"))
    width = float(element.get("width"))
    height = float(element.get("height"))

    return (
        x,
        y,
        x + width,
        y + height,
    )


def get_occluders(target):
    if target is None:
        return set()

    result = set()

    for region in target.findall(
        "./occlusion/region_overlap"
    ):
        value = region.get("occlusion_id")

        if value is not None:
            try:
                result.add(int(value))
            except ValueError:
                pass

    return result


def fit_image(image, size):
    """
    Resize while preserving aspect ratio.
    Returns an image centered on a white canvas.
    """

    result = Image.new("RGB", size, "white")

    image.thumbnail(size, Image.Resampling.LANCZOS)

    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2

    result.paste(image, (x, y))

    return result


def draw_annotations(
    image,
    frame_targets,
    target_id,
    nearby_ids,
):
    """
    Red: selected target
    Orange: annotated occluding vehicles
    Blue: nearby same-type vehicles
    """

    image = image.copy()
    draw = ImageDraw.Draw(image)

    target = frame_targets.get(target_id)

    occluders = get_occluders(target)

    # Draw nearby candidate distractors first.
    for other_id in nearby_ids:
        other = frame_targets.get(other_id)

        box = get_box(other)

        if box is None:
            continue

        draw.rectangle(
            box,
            outline="#1676D2",
            width=3,
        )

        draw.text(
            (box[0], max(0, box[1] - 15)),
            f"Nearby ID {other_id}",
            fill="#1676D2",
        )

    # Highlight annotated occluders.
    for other_id in occluders:
        if other_id == 0:
            continue

        other = frame_targets.get(other_id)

        box = get_box(other)

        if box is None:
            continue

        draw.rectangle(
            box,
            outline="#FF9800",
            width=4,
        )

        draw.text(
            (box[0], max(0, box[1] - 18)),
            f"Occluder {other_id}",
            fill="#FF9800",
        )

    # Draw selected target last.
    target_box = get_box(target)

    if target_box is not None:
        draw.rectangle(
            target_box,
            outline="#FF0000",
            width=4,
        )

        draw.text(
            (
                target_box[0],
                max(0, target_box[1] - 20),
            ),
            f"TARGET {target_id}",
            fill="#FF0000",
        )

    return image


def make_context_crop(
    image,
    target,
    other_targets=None,
):
    """
    Crop around the target with surrounding context.
    Enlarged using resizing only; no new image detail
    is introduced.
    """

    box = get_box(target)

    if box is None:
        return fit_image(image.copy(), CROP_SIZE)

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    padding = max(
        width,
        height,
    ) * CROP_PADDING

    crop_x1 = x1 - padding
    crop_y1 = y1 - padding
    crop_x2 = x2 + padding
    crop_y2 = y2 + padding

    # Also include overlapping occluder boxes where available.
    if other_targets:
        for other in other_targets:
            other_box = get_box(other)

            if other_box is None:
                continue

            ox1, oy1, ox2, oy2 = other_box

            crop_x1 = min(crop_x1, ox1)
            crop_y1 = min(crop_y1, oy1)
            crop_x2 = max(crop_x2, ox2)
            crop_y2 = max(crop_y2, oy2)

    crop_box = (
        max(0, int(crop_x1)),
        max(0, int(crop_y1)),
        min(image.width, math.ceil(crop_x2)),
        min(image.height, math.ceil(crop_y2)),
    )

    if (
        crop_box[2] <= crop_box[0]
        or crop_box[3] <= crop_box[1]
    ):
        return fit_image(image.copy(), CROP_SIZE)

    cropped = image.crop(crop_box)

    return fit_image(cropped, CROP_SIZE)


def make_panel(
    image,
    frame_number,
    label,
    target_id,
    annotations,
    nearby_ids,
):
    frame_targets = annotations.get(
        frame_number, {}
    )

    target = frame_targets.get(target_id)

    annotated = draw_annotations(
        image,
        frame_targets,
        target_id,
        nearby_ids,
    )

    full_frame = fit_image(
        annotated,
        FULL_FRAME_SIZE,
    )

    occluder_ids = get_occluders(target)

    relevant_others = [
        frame_targets[other_id]
        for other_id in occluder_ids
        if other_id in frame_targets
        and other_id != 0
    ]

    # Crop the annotated image so box labels remain visible.
    crop = make_context_crop(
        annotated,
        target,
        relevant_others,
    )

    panel = Image.new(
        "RGB",
        (PANEL_WIDTH, PANEL_HEIGHT + 180),
        "white",
    )

    draw = ImageDraw.Draw(panel)

    draw.text(
        (10, 10),
        f"{label} | Frame {frame_number}",
        fill="black",
    )

    panel.paste(
        full_frame,
        (10, 40),
    )

    draw.text(
        (10, 375),
        "Enlarged target + surroundings",
        fill="black",
    )

    panel.paste(
        crop,
        (10, 400),
    )

    if target is None:
        status = "Target has no annotation"
    elif target.find("occlusion") is not None:
        status = "Annotated occlusion"
    else:
        status = "No annotated occlusion"

    draw.text(
        (10, PANEL_HEIGHT + 165),
        status,
        fill="black",
    )

    return panel


def choose_frames(event):
    start = int(event["occlusion_start"])
    end = int(event["occlusion_end"])

    peak = int(event["peak_occlusion_frame"])

    before = event["selection_frame"]
    after = event["candidate_reappearance_frame"]

    result = []

    if before:
        result.append(
            ("Before", int(before))
        )

    result.extend([
        ("Occlusion begins", start),
        ("During", (start + peak) // 2),
        ("Maximum overlap", peak),
        ("Occlusion ends", end),
    ])

    if after:
        after = int(after)

        result.extend([
            ("First clear annotation", after),
            ("Later", after + 15),
        ])

    # Remove duplicate frame numbers.
    seen = set()
    unique = []

    for label, frame in result:
        if frame not in seen:
            unique.append((label, frame))
            seen.add(frame)

    return unique


def load_shortlisted_events(sequence):
    csv_path = (
        CSV_DIR / f"{sequence}_shortlist.csv"
    )

    with csv_path.open(newline="") as file:
        return list(csv.DictReader(file))


def create_contact_sheet(
    sequence,
    event,
    annotations,
    image_paths,
):
    target_id = int(event["target_id"])
    start = int(event["occlusion_start"])
    end = int(event["occlusion_end"])

    nearby_ids = {
        int(value)
        for value in event[
            "nearby_same_type_ids"
        ].split(",")
        if value.strip().isdigit()
    }

    selected_frames = choose_frames(event)

    panels = []

    for label, frame_num in selected_frames:
        path = image_paths.get(frame_num)

        if path is None:
            print(
                f"Missing image: {sequence}, "
                f"frame {frame_num}"
            )
            continue

        with Image.open(path) as source:
            image = source.convert("RGB")

        panel = make_panel(
            image,
            frame_num,
            label,
            target_id,
            annotations,
            nearby_ids,
        )

        panels.append(panel)

    if not panels:
        return

    columns = 2
    rows = math.ceil(
        len(panels) / columns
    )

    panel_height = PANEL_HEIGHT + 180

    sheet = Image.new(
        "RGB",
        (
            columns * PANEL_WIDTH,
            rows * panel_height,
        ),
        "#E6E6E6",
    )

    for index, panel in enumerate(panels):
        x = (index % columns) * PANEL_WIDTH
        y = (index // columns) * panel_height

        sheet.paste(panel, (x, y))

    output_folder = OUTPUT_DIR / sequence
    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"target_{target_id}_"
        f"{start}_{end}_detailed.jpg"
    )

    output_path = (
        output_folder / filename
    )

    sheet.save(
        output_path,
        quality=95,
    )

    print(f"Saved: {output_path}")


def main():
    total = 0

    for sequence, selected_ids in EVENTS.items():
        print(f"\nProcessing {sequence}")

        annotations = load_annotations(
            sequence
        )

        image_paths = load_image_paths(
            sequence
        )

        events = load_shortlisted_events(
            sequence
        )

        for target_id in selected_ids:
            expected = EXPECTED_INTERVALS[
                (sequence, target_id)
            ]

            matches = [
                event
                for event in events
                if int(event["target_id"])
                == target_id
                and (
                    int(event["occlusion_start"]),
                    int(event["occlusion_end"]),
                ) == expected
            ]

            if not matches:
                print(
                    f"Event missing: {sequence}, "
                    f"target {target_id}"
                )
                continue

            create_contact_sheet(
                sequence,
                matches[0],
                annotations,
                image_paths,
            )

            total += 1

    print(
        f"\nFinished. Generated "
        f"{total} contact sheets."
    )


if __name__ == "__main__":
    main()
