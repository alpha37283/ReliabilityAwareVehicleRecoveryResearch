from pathlib import Path
import csv
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]

EVENT_LOG = (
    ROOT
    / "data"
    / "processed"
    / "pilot_events"
    / "pilot_event_log.csv"
)

DATASET_DIR = (
    ROOT
    / "data"
    / "raw"
    / "UA-DETRAC"
)

CONTEXT_FRAMES = 30


def read_event_log():
    """
    Read the frozen Phase-0 event log.

    The final 'notes' field may itself contain commas.
    Instead of modifying the frozen CSV, any extra fields
    are joined back into the notes column.
    """

    with EVENT_LOG.open("r", newline="") as file:
        reader = csv.reader(file)

        rows = [
            row
            for row in reader
            if row
        ]

    if not rows:
        raise ValueError(
            f"Event log is empty: {EVENT_LOG}"
        )

    header = rows[0]

    expected_columns = len(header)

    events = []

    for row in rows[1:]:

        if len(row) < expected_columns:
            row += [""] * (
                expected_columns - len(row)
            )

        elif len(row) > expected_columns:
            # First columns are fixed.
            # Anything extra belongs to notes.
            fixed = row[:expected_columns - 1]

            notes = ",".join(
                row[expected_columns - 1:]
            ).strip()

            row = fixed + [notes]

        event = dict(zip(header, row))

        events.append(event)

    return events


def locate_xml(sequence):
    """
    Find the correct UA-DETRAC XML file while keeping
    train/test annotation folders separate.
    """

    candidates = [
        (
            "test",
            DATASET_DIR
            / "annotations"
            / "test_xml"
            / "DETRAC-Test-Annotations-XML"
            / f"{sequence}.xml",
        ),
        (
            "train",
            DATASET_DIR
            / "annotations"
            / "train_xml"
            / "DETRAC-Train-Annotations-XML"
            / f"{sequence}.xml",
        ),
    ]

    for split, path in candidates:
        if path.is_file():
            return path, split

    raise FileNotFoundError(
        f"No XML annotation found for {sequence}"
    )


def derive_selection_frame(
    xml_path,
    target_id,
    occlusion_start,
    lookback_frames=CONTEXT_FRAMES,
):
    """
    Find the latest non-occluded annotated frame for the
    selected target within lookback_frames before the
    manually recorded occlusion start.

    Ground truth is used here only to simulate the user's
    initial target selection.
    """

    root = ET.parse(xml_path).getroot()

    target_id = str(target_id)

    earliest_frame = max(
        1,
        occlusion_start - lookback_frames,
    )

    candidates = []

    for frame in root.findall("frame"):

        frame_num = int(frame.get("num"))

        if not (
            earliest_frame
            <= frame_num
            < occlusion_start
        ):
            continue

        for target in frame.findall(
            "./target_list/target"
        ):

            if target.get("id") != target_id:
                continue

            box = target.find("box")

            if box is None:
                continue

            has_occlusion = (
                target.find("occlusion")
                is not None
            )

            if not has_occlusion:
                candidates.append(frame_num)

    if not candidates:
        raise ValueError(
            f"No clear selection frame found for "
            f"target {target_id} before frame "
            f"{occlusion_start}"
        )

    return max(candidates)


def get_target_box(
    xml_path,
    target_id,
    frame_number,
):
    """
    Read the target bounding box at the initialization
    frame.

    This box is permitted for tracker initialization only.
    """

    root = ET.parse(xml_path).getroot()

    target_id = str(target_id)

    for frame in root.findall("frame"):

        if int(frame.get("num")) != frame_number:
            continue

        for target in frame.findall(
            "./target_list/target"
        ):

            if target.get("id") != target_id:
                continue

            box = target.find("box")

            if box is None:
                return None

            left = float(box.get("left"))
            top = float(box.get("top"))
            width = float(box.get("width"))
            height = float(box.get("height"))

            return {
                "x": left,
                "y": top,
                "width": width,
                "height": height,
                "xyxy": [
                    left,
                    top,
                    left + width,
                    top + height,
                ],
            }

    return None


def load_event(event_id):
    """
    Load one frozen Phase-0 event and resolve all paths
    needed by the future Phase-1 runner.
    """

    events = read_event_log()

    matches = [
        event
        for event in events
        if event["event_id"] == event_id
    ]

    if not matches:
        raise ValueError(
            f"Unknown event ID: {event_id}"
        )

    event = matches[0]

    sequence = event["sequence"]
    target_id = int(event["target_id"])

    occlusion_start = int(
        event["occlusion_start"]
    )

    xml_path, dataset_split = locate_xml(
        sequence
    )

    image_dir = (
        DATASET_DIR
        / "images"
        / sequence
    )

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: "
            f"{image_dir}"
        )

    selection_lookback = (
        120 if event_id == "E003"
        else CONTEXT_FRAMES
    )

    selection_frame = derive_selection_frame(
        xml_path=xml_path,
        target_id=target_id,
        occlusion_start=occlusion_start,
        lookback_frames=selection_lookback,
    )

    selection_box = get_target_box(
        xml_path=xml_path,
        target_id=target_id,
        frame_number=selection_frame,
    )

    if selection_box is None:
        raise ValueError(
            f"Could not obtain target box for "
            f"{event_id} at frame "
            f"{selection_frame}"
        )

    def optional_int(value):
        value = value.strip()

        if not value:
            return None

        return int(value)

    result = {
        "event_id": event["event_id"],
        "dataset": event["dataset"],
        "dataset_split": dataset_split,

        "sequence": sequence,
        "target_id": target_id,

        "scenario": event["scenario"],

        "selection_frame": selection_frame,
        "selection_box": selection_box,

        "occlusion_start": occlusion_start,

        "maximum_occlusion_start": optional_int(
            event["maximum_occlusion_start"]
        ),

        "reappearance_frame": optional_int(
            event["reappearance_frame"]
        ),

        "fully_visible_frame": optional_int(
            event["fully_visible_frame"]
        ),

        "complete_disappearance":
            event["complete_disappearance"],

        "similar_distractors":
            event["similar_distractors"],

        "verification_status":
            event["verification_status"],

        "notes": event["notes"],

        "image_dir": image_dir,
        "xml_path": xml_path,
    }

    return result


if __name__ == "__main__":

    event = load_event("E011")

    print("\nLoaded Phase-1 event")
    print("-" * 60)

    for key, value in event.items():
        print(f"{key}: {value}")