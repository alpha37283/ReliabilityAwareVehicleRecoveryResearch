import csv
from pathlib import Path

from evaluation.event_loader import load_event


ROOT = Path(__file__).resolve().parents[2]

OUTPUT = (
    ROOT
    / "results/phase1/phase1i_failure_taxonomy.csv"
)

EVENT_IDS = [
    f"E{i:03d}"
    for i in range(1, 13)
]

TRACKERS = [
    "sam21",
    "dam4sam",
]


FIELDS = [
    "event_id",
    "sequence",
    "target_id",
    "scenario",
    "tracker",

    # --------------------------------------------------
    # Tracking / loss stage
    # --------------------------------------------------
    "tracker_drift_before_occlusion",
    "target_completely_lost",
    "tracker_jumped_to_distractor",

    # --------------------------------------------------
    # Return / candidate generation stage
    # --------------------------------------------------
    "target_returned",
    "target_detected_after_return",

    # --------------------------------------------------
    # Identity stage
    # --------------------------------------------------
    "target_selected_rank1",
    "memory_contaminated",
    "genuine_candidate_ambiguity",

    # --------------------------------------------------
    # Failure taxonomy
    # --------------------------------------------------
    "primary_failure",
    "secondary_failures",

    # --------------------------------------------------
    # Evidence / review
    # --------------------------------------------------
    "evidence_source",
    "evidence_notes",
    "manual_review_status",
]


def unknown():
    return "unknown"


def main():

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for event_id in EVENT_IDS:

        event = load_event(event_id)

        for tracker in TRACKERS:

            rows.append({
                "event_id":
                    event_id,

                "sequence":
                    event["sequence"],

                "target_id":
                    event["target_id"],

                "scenario":
                    event.get(
                        "scenario",
                        "",
                    ),

                "tracker":
                    tracker,

                "tracker_drift_before_occlusion":
                    unknown(),

                "target_completely_lost":
                    unknown(),

                "tracker_jumped_to_distractor":
                    unknown(),

                "target_returned":
                    unknown(),

                "target_detected_after_return":
                    unknown(),

                "target_selected_rank1":
                    unknown(),

                "memory_contaminated":
                    unknown(),

                "genuine_candidate_ambiguity":
                    unknown(),

                "primary_failure":
                    "",

                "secondary_failures":
                    "",

                "evidence_source":
                    "",

                "evidence_notes":
                    "",

                "manual_review_status":
                    "pending",
            })

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "PHASE 1I — FAILURE TAXONOMY SKELETON"
    )

    print("=" * 80)

    print(
        f"Events: {len(EVENT_IDS)}"
    )

    print(
        f"Trackers: {len(TRACKERS)}"
    )

    print(
        f"Rows created: {len(rows)}"
    )

    print()

    print(
        "Failure definitions:"
    )

    print(
        "F1 — tracker drifted before occlusion"
    )

    print(
        "F2 — target completely lost"
    )

    print(
        "F3 — tracker jumped to similar distractor"
    )

    print(
        "F4 — true target returned but was not detected"
    )

    print(
        "F5 — true target was detected but not selected"
    )

    print(
        "F6 — target memory became contaminated"
    )

    print(
        "F7 — multiple candidates remained genuinely ambiguous"
    )

    print(
        "F8 — target never returned during observation"
    )

    print()

    print(
        "No failures have been assigned yet."
    )

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
