
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data/raw/UA-DETRAC"

OUTPUT_DIR = ROOT / "data/processed/pilot_events"

# Initial screening thresholds; adjust after visual inspection.
MIN_DURATION = 10
MAX_DURATION = 150
MIN_MAX_OCCLUSION = 0.50
MIN_TARGET_WIDTH = 30
MIN_TARGET_HEIGHT = 20
CONTEXT_FRAMES = 30
DISTRACTOR_DISTANCE = 3.0


def locate_xml(sequence):
    """Find the sequence XML without mixing training and test splits."""

    for split, folder in [
        ("train", "train_xml/DETRAC-Train-Annotations-XML"),
        ("test", "test_xml/DETRAC-Test-Annotations-XML"),
    ]:
        path = DATASET / "annotations" / folder / f"{sequence}.xml"

        if path.is_file():
            return path, split

    raise FileNotFoundError(
        f"Could not find annotations for {sequence}"
    )


def intersection(rect_a, rect_b):
    """Intersection of two rectangles in xyxy format."""

    x1 = max(rect_a[0], rect_b[0])
    y1 = max(rect_a[1], rect_b[1])
    x2 = min(rect_a[2], rect_b[2])
    y2 = min(rect_a[3], rect_b[3])

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def rectangle_union_area(rectangles):
    """
    Exact union area of axis-aligned rectangles.

    Prevents double-counting where multiple occlusion
    regions overlap.
    """

    if not rectangles:
        return 0.0

    xs = sorted(
        {x for r in rectangles for x in (r[0], r[2])}
    )

    total = 0.0

    for left, right in zip(xs[:-1], xs[1:]):
        intervals = []

        for x1, y1, x2, y2 in rectangles:
            if x1 < right and x2 > left:
                intervals.append((y1, y2))

        if not intervals:
            continue

        intervals.sort()

        merged_height = 0.0
        start, end = intervals[0]

        for y1, y2 in intervals[1:]:
            if y1 <= end:
                end = max(end, y2)
            else:
                merged_height += end - start
                start, end = y1, y2

        merged_height += end - start
        total += (right - left) * merged_height

    return total


def parse_box(element):
    x = float(element.get("left"))
    y = float(element.get("top"))
    w = float(element.get("width"))
    h = float(element.get("height"))

    return (x, y, x + w, y + h)


def parse_annotations(xml_path):
    """
    Returns:
      tracks[target_id][frame_number] = annotation
      frames[frame_number][target_id] = annotation
    """

    root = ET.parse(xml_path).getroot()

    tracks = defaultdict(dict)
    frames = defaultdict(dict)

    for frame in root.findall("frame"):
        frame_num = int(frame.get("num"))

        for target in frame.findall("./target_list/target"):
            target_id = int(target.get("id"))

            box_element = target.find("box")
            attr_element = target.find("attribute")

            if box_element is None:
                continue

            box = parse_box(box_element)

            x1, y1, x2, y2 = box
            target_area = (x2 - x1) * (y2 - y1)

            clipped_regions = []
            occluder_ids = set()

            for region in target.findall(
                "./occlusion/region_overlap"
            ):
                occlusion_box = parse_box(region)

                clipped = intersection(box, occlusion_box)

                if clipped is not None:
                    clipped_regions.append(clipped)

                occluder = region.get("occlusion_id")

                if occluder is not None:
                    occluder_ids.add(occluder)

            occluded_area = rectangle_union_area(
                clipped_regions
            )

            ratio = (
                occluded_area / target_area
                if target_area > 0 else 0.0
            )

            annotation = {
                "box": box,
                "width": x2 - x1,
                "height": y2 - y1,
                "vehicle_type": (
                    attr_element.get("vehicle_type", "unknown")
                    if attr_element is not None
                    else "unknown"
                ),
                "annotated_occlusion": bool(
                    target.find("occlusion") is not None
                ),
                "occlusion_ratio": min(1.0, ratio),
                "occluder_ids": occluder_ids,
            }

            tracks[target_id][frame_num] = annotation
            frames[frame_num][target_id] = annotation

    return tracks, frames


def nearby_distractors(target_id, frame_num, frames):
    """
    Find nearby vehicles with the same annotated vehicle type.

    This is only a proxy for appearance similarity.
    UA-DETRAC XML does not establish color or model identity.
    """

    frame = frames.get(frame_num, {})
    target = frame.get(target_id)

    if target is None:
        return []

    tx1, ty1, tx2, ty2 = target["box"]

    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2

    result = []

    for other_id, other in frame.items():
        if other_id == target_id:
            continue

        if other["vehicle_type"] != target["vehicle_type"]:
            continue

        if target["vehicle_type"] == "unknown":
            continue

        ox1, oy1, ox2, oy2 = other["box"]

        ocx = (ox1 + ox2) / 2
        ocy = (oy1 + oy2) / 2

        dx = tcx - ocx
        dy = tcy - ocy

        distance = (dx * dx + dy * dy) ** 0.5

        scale = max(
            target["width"],
            target["height"],
            other["width"],
            other["height"],
            1.0,
        )

        if distance / scale <= DISTRACTOR_DISTANCE:
            result.append(other_id)

    return sorted(result)


def extract_events(sequence, split, tracks, frames):
    events = []

    for target_id, observations in tracks.items():
        frame_numbers = sorted(observations)

        # Separate consecutive annotated occlusion intervals.
        intervals = []
        start = None
        previous = None

        for frame_num in frame_numbers:
            annotated = observations[frame_num][
                "annotated_occlusion"
            ]

            if annotated:
                if start is None:
                    start = frame_num

                elif previous is not None and frame_num != previous + 1:
                    intervals.append((start, previous))
                    start = frame_num

                previous = frame_num

            elif start is not None:
                intervals.append((start, previous))
                start = None
                previous = None

        if start is not None:
            intervals.append((start, previous))

        for start, end in intervals:
            interval_frames = list(range(start, end + 1))

            # Reject internally incomplete annotation intervals.
            if not all(
                f in observations for f in interval_frames
            ):
                continue

            before = [
                f for f in range(
                    max(1, start - CONTEXT_FRAMES),
                    start,
                )
                if f in observations
                and not observations[f]["annotated_occlusion"]
            ]

            after = [
                f for f in range(
                    end + 1,
                    end + CONTEXT_FRAMES + 1,
                )
                if f in observations
                and not observations[f]["annotated_occlusion"]
            ]

            selection_frame = before[-1] if before else None
            next_clear_frame = after[0] if after else None

            ratios = [
                observations[f]["occlusion_ratio"]
                for f in interval_frames
            ]

            peak_index = max(
                range(len(ratios)),
                key=lambda i: ratios[i],
            )

            peak_frame = interval_frames[peak_index]

            representative = observations[peak_frame]

            occluders = set()

            for f in interval_frames:
                occluders.update(
                    observations[f]["occluder_ids"]
                )

            # Consider distractors at the peak and first clear frame.
            distractors = set(
                nearby_distractors(
                    target_id, peak_frame, frames
                )
            )

            if next_clear_frame is not None:
                distractors.update(
                    nearby_distractors(
                        target_id, next_clear_frame, frames
                    )
                )

            duration = end - start + 1
            max_ratio = max(ratios)
            mean_ratio = sum(ratios) / len(ratios)

            suitable = (
                MIN_DURATION <= duration <= MAX_DURATION
                and max_ratio >= MIN_MAX_OCCLUSION
                and representative["width"] >= MIN_TARGET_WIDTH
                and representative["height"] >= MIN_TARGET_HEIGHT
                and selection_frame is not None
                and next_clear_frame is not None
            )

            events.append({
                "sequence": sequence,
                "dataset_split": split,
                "target_id": target_id,
                "vehicle_type": representative["vehicle_type"],
                "selection_frame": selection_frame,
                "occlusion_start": start,
                "occlusion_end": end,
                "duration_frames": duration,
                "peak_occlusion_frame": peak_frame,
                "max_occlusion_ratio": round(max_ratio, 4),
                "mean_occlusion_ratio": round(mean_ratio, 4),
                "target_width_at_peak": round(
                    representative["width"], 1
                ),
                "target_height_at_peak": round(
                    representative["height"], 1
                ),
                "candidate_reappearance_frame": next_clear_frame,
                "occluder_ids": ",".join(
                    sorted(occluders, key=lambda x: int(x)
                           if x.isdigit() else -1)
                ),
                "nearby_same_type_ids": ",".join(
                    str(x) for x in sorted(distractors)
                ),
                "nearby_same_type_count": len(distractors),
                "passes_initial_screen": suitable,
                "visual_verification": "pending",
                "confirmed_identity_loss": "unknown",
                "confirmed_recovery": "unknown",
            })

    # Screening priority, NOT a measure of actual recovery difficulty.
    events.sort(
        key=lambda e: (
            e["max_occlusion_ratio"],
            e["nearby_same_type_count"],
            e["target_width_at_peak"]
            * e["target_height_at_peak"],
        ),
        reverse=True,
    )

    return events


def save_csv(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not events:
        print(f"No events to save: {path}")
        return

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(events[0].keys()),
        )
        writer.writeheader()
        writer.writerows(events)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        default="MVI_39311",
        help="UA-DETRAC sequence name",
    )

    args = parser.parse_args()
    sequence = args.sequence

    xml_path, split = locate_xml(sequence)

    print(f"Reading: {xml_path}")
    print(f"Official split: {split}")

    tracks, frames = parse_annotations(xml_path)

    events = extract_events(
        sequence, split, tracks, frames
    )

    shortlisted = [
        e for e in events
        if e["passes_initial_screen"]
    ]

    all_path = (
        OUTPUT_DIR / f"{sequence}_all_events.csv"
    )

    shortlist_path = (
        OUTPUT_DIR / f"{sequence}_shortlist.csv"
    )

    save_csv(all_path, events)
    save_csv(shortlist_path, shortlisted)

    print(f"\nTotal annotated intervals: {len(events)}")
    print(f"Shortlisted intervals: {len(shortlisted)}")

    print("\nTop 15 shortlisted intervals:")
    print(
        "Target | Start-End | Peak ratio | "
        "Mean ratio | Same-type nearby"
    )
    print("-" * 72)

    for e in shortlisted[:15]:
        print(
            f'{e["target_id"]:>6} | '
            f'{e["occlusion_start"]:>4}-'
            f'{e["occlusion_end"]:<4} | '
            f'{e["max_occlusion_ratio"]:>10.2f} | '
            f'{e["mean_occlusion_ratio"]:>10.2f} | '
            f'{e["nearby_same_type_count"]:>16}'
        )

    print(f"\nAll events: {all_path}")
    print(f"Shortlist:  {shortlist_path}")


if __name__ == "__main__":
    main()
