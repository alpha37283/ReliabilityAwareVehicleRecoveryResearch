from pathlib import Path
import csv
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "data/raw/UA-DETRAC"
CSV_DIR = ROOT / "data/processed/pilot_events"

SEQUENCES = [
    "MVI_39311",
    "MVI_39401",
    "MVI_40761",
]

# Already manually reviewed.
EXCLUDE = {
    ("MVI_39311", 5),
    ("MVI_39401", 75),
    ("MVI_39401", 92),
    ("MVI_40761", 4),
}


def locate_xml(sequence):
    for folder in [
        "test_xml/DETRAC-Test-Annotations-XML",
        "train_xml/DETRAC-Train-Annotations-XML",
    ]:
        path = DATASET / "annotations" / folder / f"{sequence}.xml"
        if path.exists():
            return path

    raise FileNotFoundError(sequence)


def load_boxes(sequence):
    root = ET.parse(locate_xml(sequence)).getroot()

    frames = {}

    for frame in root.findall("frame"):
        frame_num = int(frame.get("num"))
        frames[frame_num] = {}

        for target in frame.findall("./target_list/target"):
            tid = int(target.get("id"))
            box = target.find("box")

            if box is None:
                continue

            x = float(box.get("left"))
            y = float(box.get("top"))
            w = float(box.get("width"))
            h = float(box.get("height"))

            frames[frame_num][tid] = {
                "cx": x + w / 2,
                "cy": y + h / 2,
                "w": w,
                "h": h,
            }

    return frames


def motion_score(frames, target_id, start_frame, end_frame):
    a = frames.get(start_frame, {}).get(target_id)
    b = frames.get(end_frame, {}).get(target_id)

    if a is None or b is None:
        return 0.0, 0.0

    distance = math.hypot(
        b["cx"] - a["cx"],
        b["cy"] - a["cy"],
    )

    # Normalize motion by approximate target size.
    scale = max(
        (a["w"] + b["w"]) / 2,
        (a["h"] + b["h"]) / 2,
        1.0,
    )

    return distance, distance / scale


all_candidates = []

for sequence in SEQUENCES:
    shortlist_path = CSV_DIR / f"{sequence}_shortlist.csv"

    with shortlist_path.open(newline="") as f:
        events = list(csv.DictReader(f))

    frames = load_boxes(sequence)

    for e in events:
        tid = int(e["target_id"])

        if (sequence, tid) in EXCLUDE:
            continue

        selection = int(e["selection_frame"])
        reappearance = int(e["candidate_reappearance_frame"])

        pixels, normalized = motion_score(
            frames,
            tid,
            selection,
            reappearance,
        )

        max_occ = float(e["max_occlusion_ratio"])
        distractors = int(e["nearby_same_type_count"])
        duration = int(e["duration_frames"])
        width = float(e["target_width_at_peak"])
        height = float(e["target_height_at_peak"])

        # Prioritize:
        # - severe occlusion
        # - target motion
        # - multiple same-type vehicles
        # - target not extremely tiny
        score = (
            max_occ * 3
            + min(normalized, 5) * 1.5
            + min(distractors, 10) * 0.35
        )

        all_candidates.append({
            "sequence": sequence,
            "target_id": tid,
            "start": int(e["occlusion_start"]),
            "end": int(e["occlusion_end"]),
            "duration": duration,
            "max_occlusion": round(max_occ, 2),
            "same_type_nearby": distractors,
            "motion_pixels": round(pixels, 1),
            "motion_normalized": round(normalized, 2),
            "target_width": round(width, 1),
            "target_height": round(height, 1),
            "priority_score": round(score, 2),
        })


# Require some movement and at least 2 same-type vehicles nearby.
filtered = [
    e for e in all_candidates
    if e["motion_normalized"] >= 0.5
    and e["same_type_nearby"] >= 2
    and e["max_occlusion"] >= 0.5
    and e["target_width"] >= 30
    and e["target_height"] >= 20
]

filtered.sort(
    key=lambda x: x["priority_score"],
    reverse=True,
)

print(f"Remaining ranked candidates: {len(filtered)}\n")

print(
    "Seq         Target   Frames       Occ   "
    "Nearby   Motion   Score"
)
print("-" * 70)

for e in filtered[:15]:
    print(
        f'{e["sequence"]:<11} '
        f'{e["target_id"]:>6}   '
        f'{e["start"]:>4}-{e["end"]:<4}   '
        f'{e["max_occlusion"]:>4.2f}   '
        f'{e["same_type_nearby"]:>6}   '
        f'{e["motion_normalized"]:>6.2f}   '
        f'{e["priority_score"]:>5.2f}'
    )

output = CSV_DIR / "next_candidate_ranking.csv"

if filtered:
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=filtered[0].keys(),
        )
        writer.writeheader()
        writer.writerows(filtered)

print(f"\nSaved: {output}")