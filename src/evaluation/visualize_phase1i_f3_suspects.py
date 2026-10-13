import csv
from pathlib import Path

from PIL import Image, ImageDraw

from evaluation.event_loader import load_event
from recovery.annotation_candidate_generator import get_frame_candidates


ROOT = Path(__file__).resolve().parents[2]

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
    / "results/phase1/phase1i_f3_review"
)

CASES = {
    "E006": [
        1080,
        1084,
        1088,
        1094,
        1100,
        1110,
        1120,
    ],

    "E012": [
        534,
        560,
        580,
        591,
        604,
        608,
    ],
}


def parse_bool(value):
    return (
        str(value)
        .strip()
        .lower()
        in {"true", "1", "yes"}
    )


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

            frame = int(
                row["frame_number"]
            )

            if not parse_bool(
                row["present"]
            ):
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

            lookup[frame] = box

    return lookup


def clamp_box(box, width, height):

    x1, y1, x2, y2 = box

    return [
        int(max(0, min(width, round(x1)))),
        int(max(0, min(height, round(y1)))),
        int(max(0, min(width, round(x2)))),
        int(max(0, min(height, round(y2)))),
    ]


def draw_label(draw, x, y, text, color):

    bbox = draw.textbbox(
        (x, y),
        text,
    )

    draw.rectangle(
        [
            bbox[0] - 2,
            bbox[1] - 2,
            bbox[2] + 2,
            bbox[3] + 2,
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

    print(
        "PHASE 1I — F3 MANUAL REVIEW VISUALIZATION"
    )
    print("=" * 100)

    for event_id, frames in CASES.items():

        event = load_event(
            event_id
        )

        sequence = event["sequence"]

        target_id = int(
            event["target_id"]
        )

        predictions = load_predictions(
            event_id
        )

        xml_path = (
            XML_ROOT
            / f"{sequence}.xml"
        )

        event_output = (
            OUTPUT_ROOT
            / event_id
        )

        event_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        images = []

        print()
        print(
            f"{event_id} | "
            f"sequence={sequence} | "
            f"target_id={target_id}"
        )

        for frame in frames:

            image_path = (
                IMAGE_ROOT
                / sequence
                / f"img{frame:05d}.jpg"
            )

            if not image_path.exists():
                print(
                    f"  frame {frame}: image missing"
                )
                continue

            image = Image.open(
                image_path
            ).convert("RGB")

            draw = ImageDraw.Draw(
                image
            )

            # ------------------------------------------
            # SAM2 prediction — RED
            # ------------------------------------------

            tracker_box = predictions.get(
                frame
            )

            if tracker_box is not None:

                tbox = clamp_box(
                    tracker_box,
                    image.width,
                    image.height,
                )

                draw.rectangle(
                    tbox,
                    outline="red",
                    width=4,
                )

                draw_label(
                    draw,
                    tbox[0],
                    max(2, tbox[1] - 18),
                    "SAM2",
                    "red",
                )

            # ------------------------------------------
            # Annotation target — GREEN
            #
            # IMPORTANT:
            # XML annotation presence does NOT prove
            # visual visibility during occlusion.
            # This is reference only.
            # ------------------------------------------

            objects = get_frame_candidates(
                str(xml_path),
                frame,
            )

            target_objects = [
                obj
                for obj in objects
                if int(
                    obj["target_id"]
                ) == target_id
            ]

            if target_objects:

                gt_box = clamp_box(
                    target_objects[0][
                        "box_xyxy"
                    ],
                    image.width,
                    image.height,
                )

                draw.rectangle(
                    gt_box,
                    outline="lime",
                    width=4,
                )

                draw_label(
                    draw,
                    gt_box[0],
                    gt_box[1] + 2,
                    "TARGET XML",
                    "lime",
                )

            # ------------------------------------------
            # Other annotated vehicles — YELLOW
            # ------------------------------------------

            for obj in objects:

                if int(
                    obj["target_id"]
                ) == target_id:
                    continue

                box = clamp_box(
                    obj["box_xyxy"],
                    image.width,
                    image.height,
                )

                draw.rectangle(
                    box,
                    outline="yellow",
                    width=2,
                )

                draw_label(
                    draw,
                    box[0],
                    box[1] + 2,
                    f"ID {obj['target_id']}",
                    "yellow",
                )

            # Frame title
            draw_label(
                draw,
                8,
                8,
                f"{event_id} frame {frame}",
                "white",
            )

            out_path = (
                event_output
                / f"{event_id}_{frame}.jpg"
            )

            image.save(
                out_path,
                quality=95,
            )

            images.append(
                image
            )

            print(
                f"  frame {frame}: "
                f"SAM2={'yes' if tracker_box is not None else 'no'}, "
                f"target_XML={'yes' if target_objects else 'no'}"
            )

        # ----------------------------------------------
        # Contact sheet
        # ----------------------------------------------

        if images:

            thumb_w = 640
            thumb_h = 360

            thumbs = []

            for image in images:

                thumb = image.copy()

                thumb.thumbnail(
                    (
                        thumb_w,
                        thumb_h,
                    )
                )

                canvas = Image.new(
                    "RGB",
                    (
                        thumb_w,
                        thumb_h,
                    ),
                    "white",
                )

                x = (
                    thumb_w
                    - thumb.width
                ) // 2

                y = (
                    thumb_h
                    - thumb.height
                ) // 2

                canvas.paste(
                    thumb,
                    (x, y),
                )

                thumbs.append(
                    canvas
                )

            cols = 2
            rows = (
                len(thumbs)
                + cols - 1
            ) // cols

            sheet = Image.new(
                "RGB",
                (
                    cols * thumb_w,
                    rows * thumb_h,
                ),
                "white",
            )

            for i, thumb in enumerate(
                thumbs
            ):

                x = (
                    i % cols
                ) * thumb_w

                y = (
                    i // cols
                ) * thumb_h

                sheet.paste(
                    thumb,
                    (x, y),
                )

            sheet_path = (
                OUTPUT_ROOT
                / f"{event_id}_contact_sheet.jpg"
            )

            sheet.save(
                sheet_path,
                quality=95,
            )

            print(
                f"  contact sheet: {sheet_path}"
            )

    print()
    print("-" * 100)

    print(
        "Legend:"
    )

    print(
        "  RED    = SAM2 prediction"
    )

    print(
        "  GREEN  = target XML box"
    )

    print(
        "  YELLOW = other annotated vehicles"
    )

    print()
    print(
        "Important: XML box presence is NOT proof "
        "that the target is visually visible."
    )

    print(
        "Use the original image content for the "
        "manual F3 judgment."
    )


if __name__ == "__main__":
    main()
