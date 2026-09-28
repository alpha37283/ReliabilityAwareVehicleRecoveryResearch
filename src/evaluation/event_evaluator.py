from pathlib import Path
import argparse
import csv
import json
import xml.etree.ElementTree as ET

from event_loader import load_event
from metrics import box_iou


ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    ROOT
    / "results"
    / "phase1"
)

# Preliminary Phase-1B evaluation threshold.
# We can later move thresholds into configs/phase1/.
TARGET_IOU_THRESHOLD = 0.30

# Require recovery to remain correct for several
# consecutive frames rather than trusting one frame.
RECOVERY_CONSECUTIVE_FRAMES = 3


# ---------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------

def load_ground_truth(xml_path):
    """
    Load all UA-DETRAC annotated boxes.

    Returns:

        ground_truth[frame_number][target_id]
            = [x1, y1, x2, y2]
    """

    root = ET.parse(xml_path).getroot()

    ground_truth = {}

    for frame in root.findall("frame"):

        frame_number = int(
            frame.get("num")
        )

        frame_targets = {}

        for target in frame.findall(
            "./target_list/target"
        ):

            target_id = int(
                target.get("id")
            )

            box_element = target.find(
                "box"
            )

            if box_element is None:
                continue

            left = float(
                box_element.get("left")
            )

            top = float(
                box_element.get("top")
            )

            width = float(
                box_element.get("width")
            )

            height = float(
                box_element.get("height")
            )

            frame_targets[target_id] = [
                left,
                top,
                left + width,
                top + height,
            ]

        ground_truth[
            frame_number
        ] = frame_targets

    return ground_truth


# ---------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------

def load_predictions(
    event_id,
    tracker_name,
):
    """
    Load predictions.csv created by
    run_phase1_event.py.
    """

    path = (
        RESULTS_DIR
        / event_id
        / tracker_name
        / "predictions.csv"
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"Prediction file not found: "
            f"{path}"
        )

    predictions = []

    with path.open(
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            present = (
                row["present"]
                .strip()
                .lower()
                == "true"
            )

            if (
                row["box_x1"] == ""
                or row["box_y1"] == ""
                or row["box_x2"] == ""
                or row["box_y2"] == ""
            ):
                box = None

            else:
                box = [
                    float(row["box_x1"]),
                    float(row["box_y1"]),
                    float(row["box_x2"]),
                    float(row["box_y2"]),
                ]

            predictions.append({
                "frame_number":
                    int(row["frame_number"]),

                "present":
                    present,

                "box":
                    box,
            })

    return predictions


# ---------------------------------------------------------
# Evaluate one frame
# ---------------------------------------------------------

def evaluate_frame(
    prediction,
    target_id,
    frame_ground_truth,
):
    """
    Compare one prediction against:
      1. the true selected target
      2. every other annotated vehicle

    This allows us to detect possible drift/switches.
    """

    predicted_box = prediction[
        "box"
    ]

    target_box = (
        frame_ground_truth.get(
            target_id
        )
    )

    target_annotated = (
        target_box is not None
    )

    target_iou = box_iou(
        predicted_box,
        target_box,
    )

    best_other_id = None
    best_other_iou = 0.0

    if predicted_box is not None:

        for other_id, other_box in (
            frame_ground_truth.items()
        ):

            if other_id == target_id:
                continue

            current_iou = box_iou(
                predicted_box,
                other_box,
            )

            if (
                current_iou
                > best_other_iou
            ):

                best_other_iou = (
                    current_iou
                )

                best_other_id = (
                    other_id
                )

    target_match = (
        prediction["present"]
        and target_annotated
        and target_iou
        >= TARGET_IOU_THRESHOLD
    )

    possible_switch = (
        prediction["present"]
        and best_other_id is not None
        and best_other_iou
        >= TARGET_IOU_THRESHOLD
        and best_other_iou
        > target_iou
    )

    return {
        "frame_number":
            prediction["frame_number"],

        "target_annotated":
            target_annotated,

        "prediction_present":
            prediction["present"],

        "target_iou":
            round(target_iou, 4),

        "target_match":
            target_match,

        "best_other_id":
            best_other_id,

        "best_other_iou":
            round(best_other_iou, 4),

        "possible_switch":
            possible_switch,
    }


# ---------------------------------------------------------
# Find first stable recovery
# ---------------------------------------------------------

def find_recovery_frame(
    evaluated_frames,
    reappearance_frame,
):
    """
    Recovery is provisional here:

    after the manually recorded reappearance frame,
    require TARGET_IOU_THRESHOLD for several
    consecutive frames.

    This is an evaluation rule only.
    """

    if reappearance_frame is None:
        return None

    post_reappearance = [
        row
        for row in evaluated_frames
        if row["frame_number"]
        >= reappearance_frame
    ]

    consecutive = 0

    first_frame = None

    for row in post_reappearance:

        if row["target_match"]:

            if consecutive == 0:
                first_frame = (
                    row["frame_number"]
                )

            consecutive += 1

            if (
                consecutive
                >= RECOVERY_CONSECUTIVE_FRAMES
            ):
                return first_frame

        else:
            consecutive = 0
            first_frame = None

    return None


# ---------------------------------------------------------
# Save frame-level evaluation
# ---------------------------------------------------------

def save_frame_evaluation(
    event_id,
    tracker_name,
    evaluated_frames,
):
    output_dir = (
        RESULTS_DIR
        / event_id
        / tracker_name
    )

    output_path = (
        output_dir
        / "evaluation_frames.csv"
    )

    if evaluated_frames:

        with output_path.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=(
                    evaluated_frames[0]
                    .keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                evaluated_frames
            )

    return output_path


# ---------------------------------------------------------
# Save event-level summary
# ---------------------------------------------------------

def save_event_summary(
    event,
    tracker_name,
    evaluated_frames,
    recovery_frame,
):
    output_dir = (
        RESULTS_DIR
        / event["event_id"]
        / tracker_name
    )

    output_path = (
        output_dir
        / "evaluation_summary.json"
    )

    switches = [
        row
        for row in evaluated_frames
        if row["possible_switch"]
    ]

    matched_frames = [
        row
        for row in evaluated_frames
        if row["target_match"]
    ]

    target_annotated_frames = [
        row
        for row in evaluated_frames
        if row["target_annotated"]
    ]

    if recovery_frame is not None:

        recovery_delay = (
            recovery_frame
            - event["reappearance_frame"]
        )

    else:
        recovery_delay = None

    summary = {
        "event_id":
            event["event_id"],

        "tracker":
            tracker_name,

        "target_id":
            event["target_id"],

        "evaluated_frames":
            len(evaluated_frames),

        "target_annotated_frames":
            len(target_annotated_frames),

        "target_matched_frames":
            len(matched_frames),

        "possible_switch_frames":
            len(switches),

        "first_possible_switch_frame":
            (
                switches[0]["frame_number"]
                if switches
                else None
            ),

        "recorded_reappearance_frame":
            event["reappearance_frame"],

        "recovery_frame":
            recovery_frame,

        "recovery_delay_frames":
            recovery_delay,

        "target_iou_threshold":
            TARGET_IOU_THRESHOLD,

        "recovery_consecutive_frames":
            RECOVERY_CONSECUTIVE_FRAMES,
    }

    with output_path.open(
        "w"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    return output_path


# ---------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------

def evaluate_event(
    event_id,
    tracker_name,
):

    event = load_event(
        event_id
    )

    predictions = load_predictions(
        event_id=event_id,
        tracker_name=tracker_name,
    )

    ground_truth = load_ground_truth(
        event["xml_path"]
    )

    evaluated_frames = []

    for prediction in predictions:

        frame_number = prediction[
            "frame_number"
        ]

        frame_ground_truth = (
            ground_truth.get(
                frame_number,
                {}
            )
        )

        result = evaluate_frame(
            prediction=prediction,
            target_id=event["target_id"],
            frame_ground_truth=(
                frame_ground_truth
            ),
        )

        evaluated_frames.append(
            result
        )

    recovery_frame = (
        find_recovery_frame(
            evaluated_frames,
            event["reappearance_frame"],
        )
    )

    frame_path = (
        save_frame_evaluation(
            event_id=event_id,
            tracker_name=tracker_name,
            evaluated_frames=(
                evaluated_frames
            ),
        )
    )

    summary_path = (
        save_event_summary(
            event=event,
            tracker_name=tracker_name,
            evaluated_frames=(
                evaluated_frames
            ),
            recovery_frame=(
                recovery_frame
            ),
        )
    )

    print(
        "\nPhase-1 Event Evaluation"
    )

    print("-" * 60)

    print(
        f"Event:            "
        f"{event_id}"
    )

    print(
        f"Tracker:          "
        f"{tracker_name}"
    )

    print(
        f"Frames evaluated: "
        f"{len(evaluated_frames)}"
    )

    print(
        f"Recovery frame:   "
        f"{recovery_frame}"
    )

    print(
        f"Frame evaluation: "
        f"{frame_path}"
    )

    print(
        f"Summary:          "
        f"{summary_path}"
    )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--event",
        required=True,
    )

    parser.add_argument(
        "--tracker",
        required=True,
    )

    args = parser.parse_args()

    evaluate_event(
        event_id=args.event,
        tracker_name=args.tracker,
    )


if __name__ == "__main__":
    main()