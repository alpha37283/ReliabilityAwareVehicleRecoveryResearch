import csv
import math
from pathlib import Path

from evaluation.event_loader import load_event


ROOT = Path(__file__).resolve().parents[2]

ELIGIBLE_FILE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

L2_FILE = (
    ROOT
    / "results/phase1/phase1f_l2_results.csv"
)


def parse_bool(value):
    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def int_or_none(value):
    value = str(value).strip()

    if not value:
        return None

    return int(float(value))


def valid_prediction(row):

    if not parse_bool(row["present"]):
        return False

    keys = [
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
    ]

    try:
        values = [
            float(row[key])
            for key in keys
        ]
    except Exception:
        return False

    if not all(
        math.isfinite(v)
        for v in values
    ):
        return False

    x1, y1, x2, y2 = values

    return (
        x2 > x1
        and y2 > y1
    )


def get_sam21_trigger_lookup():

    lookup = {}

    if not L2_FILE.exists():
        return lookup

    with L2_FILE.open(newline="") as f:

        for row in csv.DictReader(f):

            if row["tracker"] != "sam21":
                continue

            critical = int_or_none(
                row.get(
                    "persistent_critical_frame",
                    "",
                )
            )

            lost = int_or_none(
                row.get(
                    "lost_frame",
                    "",
                )
            )

            possible = [
                frame
                for frame in (critical, lost)
                if frame is not None
            ]

            lookup[row["event_id"]] = (
                min(possible)
                if possible
                else None
            )

    return lookup


def load_predictions(event_id):

    path = (
        ROOT
        / "results/phase1"
        / event_id
        / "sam21"
        / "predictions.csv"
    )

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    valid = []

    for row in rows:

        if not valid_prediction(row):
            continue

        valid.append(
            int(row["frame_number"])
        )

    return valid


def frames_before(
    frames,
    selection_frame,
    cutoff,
):

    if cutoff is None:
        return []

    return [
        frame
        for frame in frames
        if (
            frame >= selection_frame
            and frame < cutoff
        )
    ]


def last_four(frames):
    return frames[-4:]


def format_frames(frames):

    if not frames:
        return "-"

    return ",".join(
        str(frame)
        for frame in frames
    )


def main():

    trigger_lookup = (
        get_sam21_trigger_lookup()
    )

    with ELIGIBLE_FILE.open(newline="") as f:
        eligible = list(csv.DictReader(f))

    print(
        "PHASE 1H-1 — SAM2 MEMORY AVAILABILITY AUDIT"
    )
    print("=" * 110)

    print(
        "event | select | occ_start | trigger | reappear | "
        "pre-occ(valid,last4) | pre-trigger(valid,last4) | "
        "pre-reappear(valid,last4)"
    )

    print("-" * 110)

    for item in eligible:

        event_id = item["event_id"]

        event = load_event(event_id)

        selection = int(
            event["selection_frame"]
        )

        occlusion = int_or_none(
            event.get(
                "occlusion_start",
                "",
            )
        )

        reappearance = int_or_none(
            event.get(
                "reappearance_frame",
                "",
            )
        )

        trigger = trigger_lookup.get(
            event_id
        )

        valid_frames = load_predictions(
            event_id
        )

        pre_occ = frames_before(
            valid_frames,
            selection,
            occlusion,
        )

        pre_trigger = frames_before(
            valid_frames,
            selection,
            trigger,
        )

        pre_reappear = frames_before(
            valid_frames,
            selection,
            reappearance,
        )

        print(
            f"{event_id:4} | "
            f"{selection:6} | "
            f"{str(occlusion):9} | "
            f"{str(trigger):7} | "
            f"{str(reappearance):8} | "
            f"{len(pre_occ):3},"
            f"{format_frames(last_four(pre_occ)):17} | "
            f"{len(pre_trigger):3},"
            f"{format_frames(last_four(pre_trigger)):17} | "
            f"{len(pre_reappear):3},"
            f"{format_frames(last_four(pre_reappear))}"
        )


if __name__ == "__main__":
    main()
