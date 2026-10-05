import csv
from pathlib import Path

from evaluation.event_loader import load_event
from recovery.recent_memory import load_recent_memory


ROOT = Path(__file__).resolve().parents[2]

ELIGIBLE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

IMAGE_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/images"
)


def get_xyxy(box):
    if isinstance(box, dict):
        return box["xyxy"]
    return box


def main():

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    print("PHASE 1H-1 — M1 MEMORY BANK AUDIT")
    print("=" * 80)

    for row in events:

        event_id = row["event_id"]

        event = load_event(event_id)

        sequence = event["sequence"]

        predictions = (
            ROOT
            / "results/phase1"
            / event_id
            / "sam21"
            / "predictions.csv"
        )

        memory = load_recent_memory(
            predictions_csv=predictions,
            image_dir=IMAGE_ROOT / sequence,
            selection_frame=int(
                event["selection_frame"]
            ),
            memory_cutoff_frame=int(
                event["occlusion_start"]
            ),
            selection_box=get_xyxy(
                event["selection_box"]
            ),
            capacity=5,
        )

        frames = [
            item["frame_number"]
            for item in memory
        ]

        sources = [
            item["source"]
            for item in memory
        ]

        print(
            f"{event_id} | "
            f"size={len(memory)} | "
            f"frames={frames} | "
            f"sources={sources}"
        )


if __name__ == "__main__":
    main()
