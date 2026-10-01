from pathlib import Path
import argparse
import csv
import json
import time

from PIL import Image

from evaluation.event_loader import load_event
from tracking.debug_static_tracker import DebugStaticTracker


# ---------------------------------------------------------
# Project paths / settings
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "results"
    / "phase1"
)

DEFAULT_POST_REAPPEARANCE_FRAMES = 30


# ---------------------------------------------------------
# Image loading
# ---------------------------------------------------------

def load_image_paths(image_dir):
    """
    Map UA-DETRAC frame numbers to image paths.

    Example:
        img00001.jpg -> 1
        img00378.jpg -> 378
    """

    image_dir = Path(image_dir)

    result = {}

    for path in image_dir.iterdir():

        if path.suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
        }:
            continue

        digits = "".join(
            character
            for character in path.stem
            if character.isdigit()
        )

        if digits:
            result[int(digits)] = path

    if not result:
        raise RuntimeError(
            f"No images found in {image_dir}"
        )

    return result


# ---------------------------------------------------------
# Tracker selection
# ---------------------------------------------------------

def build_tracker(name, video_frame_paths=None):
    """Create one Phase-1 tracker (load SAM2 only when selected)."""

    if name == "debug":
        return DebugStaticTracker()

    if name == "sam21":
        from tracking.sam21_tracker import SAM21Tracker
        return SAM21Tracker(video_frame_paths=video_frame_paths)

    if name == "dam4sam":
        from tracking.dam4sam_tracker import DAM4SAMAdapter
        return DAM4SAMAdapter()

    raise ValueError(
        f"Unknown tracker: {name}"
    )


# ---------------------------------------------------------
# Prediction validation
# ---------------------------------------------------------

def validate_prediction(prediction):
    """
    Make sure every tracker follows our common interface.
    """

    if not isinstance(prediction, dict):
        raise TypeError(
            "Tracker prediction must be a dictionary."
        )

    required = {
        "box",
        "mask",
        "confidence",
        "present",
    }

    missing = required - prediction.keys()

    if missing:
        raise ValueError(
            f"Tracker prediction missing fields: "
            f"{sorted(missing)}"
        )


# ---------------------------------------------------------
# Choose experiment ending frame
# ---------------------------------------------------------

def choose_end_frame(
    event,
    image_paths,
    requested_end_frame=None,
):
    """
    Explicit --end-frame has priority.

    Otherwise:
        reappearance + 30 frames

    If the event has no recorded reappearance,
    run until sequence end.
    """

    sequence_end = max(image_paths)

    if requested_end_frame is not None:
        return min(
            requested_end_frame,
            sequence_end,
        )

    reappearance = event[
        "reappearance_frame"
    ]

    if reappearance is not None:
        return min(
            reappearance
            + DEFAULT_POST_REAPPEARANCE_FRAMES,
            sequence_end,
        )

    return sequence_end


# ---------------------------------------------------------
# Convert tracker output into CSV row
# ---------------------------------------------------------

def make_prediction_record(
    event,
    tracker_name,
    frame_number,
    prediction,
    runtime_ms,
):
    """
    Flatten one frame prediction so it can be
    stored in predictions.csv.
    """

    box = prediction["box"]

    if box is None:
        x1 = None
        y1 = None
        x2 = None
        y2 = None

    else:
        x1, y1, x2, y2 = box

    return {
        "event_id":
            event["event_id"],

        "sequence":
            event["sequence"],

        "tracker":
            tracker_name,

        "frame_number":
            frame_number,

        "present":
            prediction["present"],

        "box_x1":
            x1,

        "box_y1":
            y1,

        "box_x2":
            x2,

        "box_y2":
            y2,

        "confidence":
            prediction["confidence"],

        "mask_available":
            prediction["mask"] is not None,

        "runtime_ms":
            round(runtime_ms, 3),
    }


# ---------------------------------------------------------
# Save frame-by-frame predictions
# ---------------------------------------------------------

def save_predictions(
    event,
    tracker_name,
    records,
):
    """
    Save tracker output for every processed frame.
    """

    output_dir = (
        RESULTS_DIR
        / event["event_id"]
        / tracker_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "predictions.csv"
    )

    if records:

        with csv_path.open(
            "w",
            newline="",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=records[0].keys(),
            )

            writer.writeheader()

            writer.writerows(
                records
            )

    return csv_path


# ---------------------------------------------------------
# Save experiment metadata
# ---------------------------------------------------------

def save_run_metadata(
    event,
    tracker_name,
    end_frame,
    processed_frames,
    records,
):
    """
    Save information describing the experiment run.
    """

    output_dir = (
        RESULTS_DIR
        / event["event_id"]
        / tracker_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        output_dir
        / "run_metadata.json"
    )

    runtimes = [
        record["runtime_ms"]
        for record in records
    ]

    if runtimes:
        average_runtime_ms = (
            sum(runtimes)
            / len(runtimes)
        )
    else:
        average_runtime_ms = None

    metadata = {

        "event_id":
            event["event_id"],

        "dataset":
            event["dataset"],

        "dataset_split":
            event["dataset_split"],

        "sequence":
            event["sequence"],

        "target_id":
            event["target_id"],

        "scenario":
            event["scenario"],

        "tracker":
            tracker_name,

        "selection_frame":
            event["selection_frame"],

        "selection_box":
            event["selection_box"],

        "occlusion_start":
            event["occlusion_start"],

        "maximum_occlusion_start":
            event["maximum_occlusion_start"],

        "reappearance_frame":
            event["reappearance_frame"],

        "run_end_frame":
            end_frame,

        "processed_future_frames":
            processed_frames,

        "average_tracker_runtime_ms":
            (
                round(
                    average_runtime_ms,
                    3,
                )
                if average_runtime_ms
                is not None
                else None
            ),
    }

    with metadata_path.open(
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    return metadata_path


# ---------------------------------------------------------
# Main event-running function
# ---------------------------------------------------------

def run_event(
    event_id,
    tracker_name,
    requested_end_frame=None,
):

    # -----------------------------------------------------
    # Load event
    # -----------------------------------------------------

    event = load_event(
        event_id
    )

    print("\nPhase-1 Event Runner")
    print("-" * 60)

    print(
        f"Event:       "
        f"{event['event_id']}"
    )

    print(
        f"Sequence:    "
        f"{event['sequence']}"
    )

    print(
        f"Target ID:   "
        f"{event['target_id']}"
    )

    print(
        f"Tracker:     "
        f"{tracker_name}"
    )

    # -----------------------------------------------------
    # Load available sequence images
    # -----------------------------------------------------

    image_paths = load_image_paths(
        event["image_dir"]
    )

    selection_frame = event[
        "selection_frame"
    ]

    if selection_frame not in image_paths:

        raise FileNotFoundError(
            f"Selection frame "
            f"{selection_frame} not found."
        )

    end_frame = choose_end_frame(
        event=event,
        image_paths=image_paths,
        requested_end_frame=(
            requested_end_frame
        ),
    )

    print(
        f"Frame range: "
        f"{selection_frame} -> "
        f"{end_frame}"
    )

    # -----------------------------------------------------
    # Create tracker
    # -----------------------------------------------------

    tracker = build_tracker(
        tracker_name,
        video_frame_paths={
            n: image_paths[n]
            for n in sorted(image_paths)
            if selection_frame <= n <= end_frame
        },
    )

    # -----------------------------------------------------
    # Initialize tracker
    # -----------------------------------------------------

    selection_path = image_paths[
        selection_frame
    ]

    with Image.open(
        selection_path
    ) as image:

        selection_image = (
            image.convert("RGB")
        )

    target_box = event[
        "selection_box"
    ]["xyxy"]

    tracker.initialize(
        frame=selection_image,
        target_box=target_box,
    )

    print(
        f"Initialized at frame "
        f"{selection_frame}"
    )

    print(
        f"Initialization box: "
        f"{target_box}"
    )

    # -----------------------------------------------------
    # Determine future frames
    # -----------------------------------------------------

    frame_numbers = sorted(

        frame_number

        for frame_number
        in image_paths

        if (
            selection_frame
            < frame_number
            <= end_frame
        )
    )

    # Stores all frame-level results.
    records = []

    processed = 0

    # -----------------------------------------------------
    # Run tracker frame-by-frame
    # -----------------------------------------------------

    for frame_number in frame_numbers:

        frame_path = image_paths[
            frame_number
        ]

        with Image.open(
            frame_path
        ) as image:

            frame = image.convert(
                "RGB"
            )

        # ---------------------------------------------
        # Measure TRACKER runtime only.
        #
        # Image loading is intentionally outside this
        # timing block.
        # ---------------------------------------------

        # Synchronize CUDA before/after inference, if the adapter supports it.
        # Otherwise perf_counter would under-report asynchronous GPU work.
        sync = getattr(tracker, "synchronize", None)
        if callable(sync):
            sync()

        start_time = (
            time.perf_counter()
        )

        prediction = tracker.track(
            frame
        )

        if callable(sync):
            sync()

        end_time = (
            time.perf_counter()
        )

        runtime_ms = (
            end_time
            - start_time
        ) * 1000

        # ---------------------------------------------
        # Verify common tracker interface
        # ---------------------------------------------

        validate_prediction(
            prediction
        )

        # ---------------------------------------------
        # Create permanent frame record
        # ---------------------------------------------

        record = make_prediction_record(
            event=event,
            tracker_name=tracker_name,
            frame_number=frame_number,
            prediction=prediction,
            runtime_ms=runtime_ms,
        )

        records.append(
            record
        )

        processed += 1

        # ---------------------------------------------
        # Temporary terminal progress
        # ---------------------------------------------

        if (
            processed == 1
            or processed % 25 == 0
            or frame_number == end_frame
        ):

            print(
                f"Frame {frame_number}: "
                f"present="
                f"{prediction['present']}, "
                f"box="
                f"{prediction['box']}, "
                f"runtime="
                f"{runtime_ms:.3f} ms"
            )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    predictions_path = (
        save_predictions(
            event=event,
            tracker_name=tracker_name,
            records=records,
        )
    )

    metadata_path = (
        save_run_metadata(
            event=event,
            tracker_name=tracker_name,
            end_frame=end_frame,
            processed_frames=processed,
            records=records,
        )
    )

    # -----------------------------------------------------
    # Reset tracker
    # -----------------------------------------------------

    tracker.reset()

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("-" * 60)

    print(
        f"Finished event "
        f"{event_id}"
    )

    print(
        f"Processed future frames: "
        f"{processed}"
    )

    print(
        f"Predictions saved: "
        f"{predictions_path}"
    )

    print(
        f"Metadata saved: "
        f"{metadata_path}"
    )


# ---------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run one Phase-1 vehicle "
            "tracking event."
        )
    )

    parser.add_argument(
        "--event",
        required=True,
        help=(
            "Phase-0 event ID, "
            "for example E011"
        ),
    )

    parser.add_argument(
        "--tracker",
        default="debug",
        help=(
            "Tracker backend. "
            "Available: debug, sam21, dam4sam"
        ),
    )

    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help=(
            "Optional explicit "
            "last frame."
        ),
    )

    args = parser.parse_args()

    run_event(
        event_id=args.event,
        tracker_name=args.tracker,
        requested_end_frame=(
            args.end_frame
        ),
    )


if __name__ == "__main__":
    main()