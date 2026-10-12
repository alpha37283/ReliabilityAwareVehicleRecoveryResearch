import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TAXONOMY = (
    ROOT
    / "results/phase1/phase1i_failure_taxonomy.csv"
)


# ----------------------------------------------------------
# Frozen Phase 1E / manual-review facts
#
# "target_completely_lost" refers to the evaluated target
# episode, not simply whether the tracker emitted a box.
# ----------------------------------------------------------

TRUE_LOSS = {
    "E005": "yes",
    "E006": "yes",
    "E007": "yes",
    "E012": "yes",
}

NO_TRUE_LOSS = {
    "E001": "no",
    "E002": "no",
    "E003": "no",
    "E004": "no",
    "E008": "no",
    "E009": "no",
    "E011": "no",
}

UNRESOLVED_LOSS = {
    "E010": "unknown",
}


def append_text(original, text):

    original = str(original).strip()

    if not original:
        return text

    if text in original:
        return original

    return original + "; " + text


def main():

    with TAXONOMY.open(newline="") as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        rows = list(reader)

    print(
        "PHASE 1I — POPULATE TRACKING / LOSS FACTS"
    )
    print("=" * 105)

    for row in rows:

        event_id = row["event_id"]
        tracker = row["tracker"]

        # ==================================================
        # F2-related objective loss state
        # ==================================================

        if event_id in TRUE_LOSS:

            row[
                "target_completely_lost"
            ] = "yes"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1E event-aware loss evaluation"
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                "verified genuine target-loss episode"
            )

        elif event_id in NO_TRUE_LOSS:

            row[
                "target_completely_lost"
            ] = "no"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1E event-aware loss evaluation"
            )

            if event_id == "E003":

                note = (
                    "no true recovery loss in evaluated episode; "
                    "later target exit must not be counted as "
                    "occlusion-recovery loss"
                )

            else:

                note = (
                    "no verified genuine target-loss episode "
                    "in evaluated recovery interval"
                )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                note
            )

        elif event_id in UNRESOLVED_LOSS:

            row[
                "target_completely_lost"
            ] = "unknown"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase0/Phase1 manual review"
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                (
                    "E010 remained occluded / unresolved and "
                    "was not frozen as a strict verified-loss "
                    "or verified-absence case"
                )
            )

        # ==================================================
        # F3 possible jump-to-distractor cases
        #
        # Do NOT convert these to YES yet.
        # They require manual trajectory confirmation.
        # ==================================================

        if (
            event_id == "E006"
            and tracker == "sam21"
        ):

            row[
                "tracker_jumped_to_distractor"
            ] = "suspected"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1C/Phase1E tracker review"
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                (
                    "SAM2 produced wrong boxes while target "
                    "was hidden; possible distractor jump, "
                    "requires manual confirmation"
                )
            )

        elif (
            event_id == "E006"
            and tracker == "dam4sam"
        ):

            row[
                "tracker_jumped_to_distractor"
            ] = "no"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1C tracker review"
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                (
                    "DAM4SAM abstained/stopped rather than "
                    "continuing with the wrong hidden box"
                )
            )

        elif (
            event_id == "E012"
            and tracker == "sam21"
        ):

            row[
                "tracker_jumped_to_distractor"
            ] = "suspected"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1C/Phase1F tracker review"
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                (
                    "late SAM2 drift was previously suspected; "
                    "specific distractor identity has not yet "
                    "been manually verified"
                )
            )

        # F1 remains unknown for now.
        # We must explicitly inspect pre-occlusion tracking
        # before declaring drift-before-occlusion.

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    with TAXONOMY.open(
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
    # Compact event-level summary
    # --------------------------------------------------

    print(
        "Event-level genuine-loss status:"
    )
    print()

    for event_id in [
        f"E{i:03d}"
        for i in range(1, 13)
    ]:

        matching = [
            r for r in rows
            if r["event_id"] == event_id
        ]

        loss = matching[0][
            "target_completely_lost"
        ]

        print(
            f"{event_id} | "
            f"target_completely_lost={loss}"
        )

    print()
    print("-" * 105)

    yes_rows = sum(
        row["target_completely_lost"] == "yes"
        for row in rows
    )

    no_rows = sum(
        row["target_completely_lost"] == "no"
        for row in rows
    )

    unknown_rows = sum(
        row["target_completely_lost"] == "unknown"
        for row in rows
    )

    print(
        f"Tracker rows with verified loss: "
        f"{yes_rows}/24"
    )

    print(
        f"Tracker rows without verified loss: "
        f"{no_rows}/24"
    )

    print(
        f"Tracker rows unresolved: "
        f"{unknown_rows}/24"
    )

    print()

    print(
        "Potential F3 cases requiring manual confirmation:"
    )

    for row in rows:

        if (
            row["tracker_jumped_to_distractor"]
            == "suspected"
        ):

            print(
                f"  {row['event_id']} / "
                f"{row['tracker']}"
            )

    print()
    print(
        "F1 drift-before-occlusion remains UNKNOWN."
    )

    print(
        "No F1-F8 taxonomy labels have been assigned yet."
    )

    print()
    print(f"Updated: {TAXONOMY}")


if __name__ == "__main__":
    main()
