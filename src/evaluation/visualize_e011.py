from pathlib import Path
import csv
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]

SEQUENCE = "MVI_40761"
TARGET_ID = 24
FRAMES = [353, 363, 374, 378]

IMAGE_DIR = ROOT / "data/raw/UA-DETRAC/images" / SEQUENCE
XML_FILE = (
    ROOT / "data/raw/UA-DETRAC/annotations"
    / "test_xml/DETRAC-Test-Annotations-XML"
    / f"{SEQUENCE}.xml"
)
PRED_FILE = ROOT / "results/phase1/E011/sam21/predictions.csv"
OUT_DIR = ROOT / "results/visualizations/E011_sam21"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load SAM2.1 predictions
with PRED_FILE.open(newline="") as f:
    predictions = {
        int(row["frame_number"]): row
        for row in csv.DictReader(f)
    }

# Load UA-DETRAC ground truth
xml_root = ET.parse(XML_FILE).getroot()

for frame_num in FRAMES:
    image_path = IMAGE_DIR / f"img{frame_num:05d}.jpg"

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Annotated boxes (target and distractors)
    for frame_element in xml_root.iter("frame"):
        if int(frame_element.attrib["num"]) != frame_num:
            continue

        for target in frame_element.findall("./target_list/target"):
            obj_id = int(target.attrib["id"])
            box = target.find("box")
            if box is None:
                continue

            x = float(box.attrib["left"])
            y = float(box.attrib["top"])
            w = float(box.attrib["width"])
            h = float(box.attrib["height"])

            color = "blue" if obj_id == TARGET_ID else "orange"

            draw.rectangle(
                [x, y, x + w, y + h],
                outline=color,
                width=2
            )

            draw.text(
                (x, max(0, y - 13)),
                f"GT {obj_id}",
                fill=color
            )

        break

    # SAM2.1 prediction
    pred = predictions.get(frame_num)

    if pred and pred["box_x1"] not in ("", None):
        coords = [
            float(pred[f"box_{v}"])
            for v in ("x1", "y1", "x2", "y2")
        ]

        draw.rectangle(
            coords,
            outline="lime",
            width=3
        )

        draw.text(
            (coords[0], max(0, coords[1] - 25)),
            "SAM2.1",
            fill="lime"
        )

    output = OUT_DIR / f"frame_{frame_num}.jpg"
    image.save(output)

    print(f"Saved: {output}")