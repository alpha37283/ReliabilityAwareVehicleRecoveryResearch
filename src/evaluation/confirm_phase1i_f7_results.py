import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

TAXONOMY = (
    ROOT
    / "results/phase1/phase1i_failure_taxonomy.csv"
)

F7_YES = {
    "E001": (
        "Manual target-vs-top-candidate review confirmed "
        "multiple visually similar light/silver vehicles "
        "remained plausible."
    ),
    "E008": (
        "Manual review confirmed the true dark vehicle was "
        "among several closely similar high-ranked candidates."
    ),
    "E011": (
        "Manual target-vs-top-candidate review confirmed "
        "multiple taxi-like vehicles with highly similar "
        "color/livery and viewpoint remained plausible."
    ),
}

F7_NO = {
    "E003": (
        "Manual target-vs-top-candidate review showed the "
        "true target was visually distinguishable from the "
        "highest-ranked wrong candidates; failure is better "
        "classified as appearance-selection failure (F5)."
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

    print("PHASE 1I — CONFIRM F7 MANUAL REVIEW")
    print("=" * 90)

    yes_count = 0
    no_count = 0

    for row in rows:

        event_id = row["event_id"]

        if event_id in F7_YES:

            row[
                "genuine_candidate_ambiguity"
            ] = "yes"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1I manual F7 target-vs-top review",
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                F7_YES[event_id],
            )

            yes_count += 1

        elif event_id in F7_NO:

            row[
                "genuine_candidate_ambiguity"
            ] = "no"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1I manual F7 target-vs-top review",
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                F7_NO[event_id],
            )

            no_count += 1

    with TAXONOMY.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"F7 YES rows: {yes_count}/24"
    )

    print(
        f"F7 NO rows explicitly reviewed: {no_count}/24"
    )

    print()

    print(
        "Confirmed F7 events: E001, E008, E011"
    )

    print(
        "Reviewed non-F7 event: E003"
    )

    print()

    print(
        "F7 is event-level evidence and is therefore "
        "copied across both tracker rows."
    )

    print()

    print(f"Updated: {TAXONOMY}")


if __name__ == "__main__":
    main()
