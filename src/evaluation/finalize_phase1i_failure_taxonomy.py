import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "results/phase1/phase1i_failure_taxonomy.csv"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1i_failure_taxonomy_final.csv"
)

SUMMARY = (
    ROOT
    / "results/phase1/phase1i_failure_summary.csv"
)


FAILURE_NAMES = {
    "F1": "tracker drifted before occlusion",
    "F2": "target completely lost",
    "F3": "tracker jumped to similar distractor",
    "F4": "true target returned but was not detected",
    "F5": "true target detected but not selected",
    "F6": "target memory became contaminated",
    "F7": "multiple candidates remained genuinely ambiguous",
    "F8": "target never returned during observation",
}


# ----------------------------------------------------------
# Manually reviewed / frozen F7 decisions.
# ----------------------------------------------------------

F7_YES_EVENTS = {
    "E001",
    "E008",
    "E011",
}

F7_NO_EVENTS = {
    "E003",
    "E004",
}


def add_secondary(labels, code):

    if code not in labels:
        labels.append(code)


def main():

    with INPUT.open(newline="") as f:

        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(
        "PHASE 1I — FINAL FAILURE TAXONOMY"
    )
    print("=" * 115)

    # --------------------------------------------------
    # Finalize F7 manually reviewed event facts first.
    # --------------------------------------------------

    for row in rows:

        event_id = row["event_id"]

        if event_id in F7_YES_EVENTS:

            row[
                "genuine_candidate_ambiguity"
            ] = "yes"

        elif event_id in F7_NO_EVENTS:

            row[
                "genuine_candidate_ambiguity"
            ] = "no"

    # --------------------------------------------------
    # Assign primary + secondary taxonomy
    # --------------------------------------------------

    for row in rows:

        event_id = row["event_id"]
        tracker = row["tracker"]

        primary = None
        secondary = []

        drift = row[
            "tracker_drift_before_occlusion"
        ]

        lost = row[
            "target_completely_lost"
        ]

        jumped = row[
            "tracker_jumped_to_distractor"
        ]

        returned = row[
            "target_returned"
        ]

        detected = row[
            "target_detected_after_return"
        ]

        selected = row[
            "target_selected_rank1"
        ]

        contaminated = row[
            "memory_contaminated"
        ]

        ambiguous = row[
            "genuine_candidate_ambiguity"
        ]

        # ==================================================
        # E010 remains unresolved by design.
        # Do not force a failure taxonomy.
        # ==================================================

        if event_id == "E010":

            primary = "UNRESOLVED"

        # ==================================================
        # F8 — target never returned.
        #
        # F8 is an event condition rather than an
        # algorithmic identity error, but it is retained
        # in the requested taxonomy.
        # ==================================================

        elif returned == "no":

            primary = "F8"

            if lost == "yes":
                add_secondary(
                    secondary,
                    "F2",
                )

            if jumped == "yes":
                add_secondary(
                    secondary,
                    "F3",
                )

        # ==================================================
        # Returned target missed by candidate generator.
        # ==================================================

        elif (
            returned == "yes"
            and detected == "no"
        ):

            primary = "F4"

            if lost == "yes":
                add_secondary(
                    secondary,
                    "F2",
                )

            if jumped == "yes":
                add_secondary(
                    secondary,
                    "F3",
                )

        # ==================================================
        # Returned + detected but identity selection failed.
        # ==================================================

        elif (
            returned == "yes"
            and detected == "yes"
            and selected == "no"
        ):

            if ambiguous == "yes":

                primary = "F7"

                # Detection succeeded but identity ranking
                # still failed, so F5 is also true.
                add_secondary(
                    secondary,
                    "F5",
                )

            else:

                primary = "F5"

            if lost == "yes":
                add_secondary(
                    secondary,
                    "F2",
                )

            if jumped == "yes":
                add_secondary(
                    secondary,
                    "F3",
                )

            if contaminated == "yes":
                add_secondary(
                    secondary,
                    "F6",
                )

            if drift == "yes":
                add_secondary(
                    secondary,
                    "F1",
                )

        # ==================================================
        # Identity succeeded, but tracking may still have
        # experienced genuine loss / distractor jump.
        # ==================================================

        elif (
            returned == "yes"
            and detected == "yes"
            and selected == "yes"
        ):

            if lost == "yes":

                primary = "F2"

                if jumped == "yes":
                    add_secondary(
                        secondary,
                        "F3",
                    )

            elif jumped == "yes":

                primary = "F3"

            elif contaminated == "yes":

                primary = "F6"

            elif drift == "yes":

                primary = "F1"

            else:

                primary = "NONE"

        # ==================================================
        # Remaining uncertain combinations.
        # ==================================================

        else:

            primary = "UNRESOLVED"

        row[
            "primary_failure"
        ] = primary

        row[
            "secondary_failures"
        ] = ",".join(
            secondary
        )

        # Mark completed rows except deliberately unresolved.
        if primary != "UNRESOLVED":

            row[
                "manual_review_status"
            ] = "phase1i_finalized"

        print(
            f"{event_id} / {tracker:8} | "
            f"primary={primary:10} | "
            f"secondary="
            f"{','.join(secondary) if secondary else '-'}"
        )

    # --------------------------------------------------
    # Save final taxonomy
    # --------------------------------------------------

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # Primary failure counts — tracker-row level
    # --------------------------------------------------

    primary_counts = Counter(
        row["primary_failure"]
        for row in rows
    )

    # --------------------------------------------------
    # Unique event occurrence of each failure anywhere
    # in primary OR secondary labels.
    # --------------------------------------------------

    events_by_failure = defaultdict(set)

    for row in rows:

        event_id = row["event_id"]

        primary = row[
            "primary_failure"
        ]

        if primary in FAILURE_NAMES:

            events_by_failure[
                primary
            ].add(event_id)

        secondary = [
            x.strip()
            for x in row[
                "secondary_failures"
            ].split(",")
            if x.strip()
        ]

        for code in secondary:

            if code in FAILURE_NAMES:

                events_by_failure[
                    code
                ].add(event_id)

    summary_rows = []

    for code in [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
    ]:

        events = sorted(
            events_by_failure[
                code
            ]
        )

        summary_rows.append({
            "failure_code":
                code,

            "failure_name":
                FAILURE_NAMES[code],

            "unique_event_count":
                len(events),

            "events":
                ",".join(events),

            "primary_tracker_row_count":
                primary_counts.get(
                    code,
                    0,
                ),
        })

    with SUMMARY.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    # --------------------------------------------------
    # Console summaries
    # --------------------------------------------------

    print()
    print("-" * 115)

    print(
        "PRIMARY FAILURE COUNTS "
        "(24 event × tracker rows)"
    )

    for key in [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "F6",
        "F7",
        "F8",
        "NONE",
        "UNRESOLVED",
    ]:

        print(
            f"{key:10} "
            f"{primary_counts.get(key, 0):2d}"
        )

    print()
    print(
        "UNIQUE EVENTS CONTAINING EACH FAILURE "
        "(primary or secondary)"
    )

    for row in summary_rows:

        print(
            f"{row['failure_code']} | "
            f"{row['unique_event_count']}/12 | "
            f"{row['events'] if row['events'] else '-'}"
        )

    print()
    print(
        "Important interpretation:"
    )

    print(
        "  F5/F7 are identity-stage event findings and are "
        "duplicated across the two tracker rows."
    )

    print(
        "  F3 is tracker-specific where manually verified."
    )

    print(
        "  F8 represents absence of verified return rather "
        "than a recoverable identity-selection error."
    )

    print(
        "  E010 remains deliberately unresolved."
    )

    print()
    print(
        f"Saved final taxonomy: {OUTPUT}"
    )

    print(
        f"Saved summary:        {SUMMARY}"
    )


if __name__ == "__main__":
    main()
