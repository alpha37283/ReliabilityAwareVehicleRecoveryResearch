import csv
from pathlib import Path

from evaluation.event_loader import load_event
from recovery.annotation_candidate_generator import (
    get_frame_candidates,
    filter_plausible_candidates,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENT_LOG = (
    PROJECT_ROOT
    / "data/processed/pilot_events/pilot_event_log.csv"
)

XML_ROOT = (
    PROJECT_ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml/"
      "DETRAC-Test-Annotations-XML"
)

OUTPUT = (
    PROJECT_ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)


def main():
    rows = []

    event_ids = [f"E{i:03d}" for i in range(1, 13)]

    for event_id in event_ids:

        event = load_event(event_id)

        reappearance = event.get("reappearance_frame")

        # Candidate recall at reappearance is undefined if
        # there is no recorded reappearance frame.
        if reappearance in (None, "", "None"):
            print(f"{event_id}: SKIP — no recorded reappearance")
            continue

        frame = int(reappearance)
        sequence = event["sequence"]
        target_id = int(event["target_id"])
        selection_box = event["selection_box"]

        if isinstance(selection_box, dict):
            reference_box = selection_box["xyxy"]
        else:
            reference_box = selection_box

        xml_path = XML_ROOT / f"{sequence}.xml"

        all_candidates = get_frame_candidates(
            xml_path,
            frame,
        )

        plausible = filter_plausible_candidates(
            all_candidates,
            reference_box,
        )

        true_in_all = any(
            c["target_id"] == target_id
            for c in all_candidates
        )

        true_in_plausible = any(
            c["target_id"] == target_id
            for c in plausible
        )

        distractor_count = sum(
            c["target_id"] != target_id
            for c in plausible
        )

        rows.append({
            "event_id": event_id,
            "sequence": sequence,
            "target_id": target_id,
            "reappearance_frame": frame,
            "raw_candidate_count": len(all_candidates),
            "plausible_candidate_count": len(plausible),
            "distractor_count": distractor_count,
            "true_target_in_raw": true_in_all,
            "true_target_in_plausible": true_in_plausible,
        })

        print(
            f"{event_id} | frame={frame} | "
            f"raw={len(all_candidates)} | "
            f"plausible={len(plausible)} | "
            f"distractors={distractor_count} | "
            f"target_retained={true_in_plausible}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "event_id",
        "sequence",
        "target_id",
        "reappearance_frame",
        "raw_candidate_count",
        "plausible_candidate_count",
        "distractor_count",
        "true_target_in_raw",
        "true_target_in_plausible",
    ]

    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    eligible = len(rows)

    retained = sum(
        r["true_target_in_plausible"]
        for r in rows
    )

    recall = (
        retained / eligible
        if eligible > 0
        else 0.0
    )

    print()
    print(f"Eligible events: {eligible}")
    print(f"True target retained: {retained}")
    print(f"Reference candidate retention: {recall:.3f}")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
