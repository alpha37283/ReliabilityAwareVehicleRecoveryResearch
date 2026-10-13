import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TAXONOMY = (
    ROOT
    / "results/phase1/phase1i_failure_taxonomy.csv"
)


CONFIRMED_F3 = {
    ("E006", "sam21"): (
        "Manual contact-sheet review: after SAM2 absence at "
        "frames 1094-1100, the tracker resumes at frames "
        "1110-1120 on a different nearby vehicle while the "
        "true target remains elsewhere."
    ),

    ("E012", "sam21"): (
        "Manual contact-sheet review: after the target leaves/"
        "does not return, SAM2 continues at frames 604-608 on "
        "another visible vehicle."
    ),
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

    print("PHASE 1I — CONFIRM F3 MANUAL REVIEW")
    print("=" * 90)

    confirmed = 0

    for row in rows:

        key = (
            row["event_id"],
            row["tracker"],
        )

        if key not in CONFIRMED_F3:
            continue

        row[
            "tracker_jumped_to_distractor"
        ] = "yes"

        row["evidence_source"] = append_text(
            row["evidence_source"],
            "Phase1I manual F3 contact-sheet review",
        )

        row["evidence_notes"] = append_text(
            row["evidence_notes"],
            CONFIRMED_F3[key],
        )

        row[
            "manual_review_status"
        ] = "F3_reviewed"

        confirmed += 1

        print(
            f"{row['event_id']} / "
            f"{row['tracker']} -> F3 evidence CONFIRMED"
        )

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

    print()
    print(
        f"Confirmed jump-to-distractor rows: "
        f"{confirmed}"
    )

    print()
    print(
        "No primary/secondary F1-F8 labels assigned yet."
    )

    print(f"Updated: {TAXONOMY}")


if __name__ == "__main__":
    main()
