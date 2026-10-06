import csv
from pathlib import Path

from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder
from recovery.distractor_memory import build_distractor_memory


ROOT = Path(__file__).resolve().parents[2]

ELIGIBLE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

IMAGE_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/images"
)


def main():

    encoder = AppearanceEncoder()
    detector = YOLO("yolo11n.pt")

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    print("PHASE 1H-1 — M2 NEGATIVE MEMORY AUDIT")
    print("=" * 100)

    events_with_memory = 0
    total_entries = 0

    for row in events:

        event_id = row["event_id"]

        event = load_event(event_id)

        sequence = event["sequence"]

        memory = build_distractor_memory(
            predictions_csv=(
                ROOT
                / "results/phase1"
                / event_id
                / "sam21"
                / "predictions.csv"
            ),
            image_dir=(
                IMAGE_ROOT
                / sequence
            ),
            selection_frame=int(
                event["selection_frame"]
            ),
            memory_cutoff_frame=int(
                event["occlusion_start"]
            ),
            encoder=encoder,
            detector=detector,
            capacity=3,
        )

        if memory:
            events_with_memory += 1

        total_entries += len(memory)

        if not memory:

            print(
                f"{event_id} | "
                "negative_memory=0 | fallback=M1"
            )

            continue

        entries = []

        for item in memory:

            entries.append(
                (
                    f"f{item['frame_number']}:"
                    f"sim={item['target_distractor_similarity']:.3f},"
                    f"dist={item['normalized_distance']:.2f}"
                )
            )

        print(
            f"{event_id} | "
            f"negative_memory={len(memory)} | "
            + " | ".join(entries)
        )

    print()
    print("-" * 100)

    print(
        f"Events with M2 negative memory: "
        f"{events_with_memory}/{len(events)}"
    )

    print(
        f"Total negative memory entries: "
        f"{total_entries}"
    )


if __name__ == "__main__":
    main()
