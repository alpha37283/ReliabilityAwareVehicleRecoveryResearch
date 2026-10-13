import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TAXONOMY = (
    ROOT
    / "results/phase1/phase1i_failure_taxonomy.csv"
)


ASSESSABLE_NO_F1 = {
    "E003",
    "E005",
    "E006",
    "E007",
    "E008",
    "E009",
    "E010",
    "E011",
}


NOT_ASSESSABLE = {
    "E001",
    "E002",
    "E004",
    "E012",
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
        "PHASE 1I — CONFIRM F1 PRE-OCCLUSION RESULTS"
    )
    print("=" * 95)

    no_f1 = 0
    not_assessable = 0

    for row in rows:

        event_id = row["event_id"]

        if event_id in ASSESSABLE_NO_F1:

            row[
                "tracker_drift_before_occlusion"
            ] = "no"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1I pre-occlusion IoU diagnostic",
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                (
                    "no sustained pre-occlusion drift; "
                    "no IoU<0.30 tracker frames before "
                    "occlusion_start"
                ),
            )

            no_f1 += 1

        elif event_id in NOT_ASSESSABLE:

            row[
                "tracker_drift_before_occlusion"
            ] = "not_assessable"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1I pre-occlusion IoU diagnostic",
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                (
                    "occlusion begins essentially immediately "
                    "after tracker initialization; insufficient "
                    "pre-occlusion trajectory to assess F1"
                ),
            )

            not_assessable += 1

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

    print(
        f"F1 = NO: "
        f"{no_f1}/24 tracker rows"
    )

    print(
        f"F1 not assessable: "
        f"{not_assessable}/24 tracker rows"
    )

    print(
        "F1 confirmed YES: 0/24"
    )

    print()
    print(
        "Important: initialization-frame absence was "
        "not treated as tracker drift."
    )

    print()
    print(f"Updated: {TAXONOMY}")


if __name__ == "__main__":
    main()
