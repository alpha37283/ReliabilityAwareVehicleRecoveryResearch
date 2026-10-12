import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TAXONOMY = (
    ROOT
    / "results/phase1/phase1i_failure_taxonomy.csv"
)

M1_RESULTS = (
    ROOT
    / "results/phase1/phase1h1_m1_all_events.csv"
)


def parse_rank(row):
    return int(row["target_rank"])


def append_text(original, text):

    original = str(original).strip()

    if not original:
        return text

    if text in original:
        return original

    return original + "; " + text


def main():

    # --------------------------------------------------
    # Load frozen Phase 1H M1 identity results
    # --------------------------------------------------

    with M1_RESULTS.open(newline="") as f:

        m1 = {
            row["event_id"]: row
            for row in csv.DictReader(f)
        }

    # --------------------------------------------------
    # Load Phase 1I skeleton
    # --------------------------------------------------

    with TAXONOMY.open(newline="") as f:

        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames
        rows = list(reader)

    print(
        "PHASE 1I — POPULATE OBJECTIVE FACTS"
    )
    print("=" * 95)

    for row in rows:

        event_id = row["event_id"]

        # ==================================================
        # 10 events with verified reappearance + Phase 1G
        # candidate retention + Phase 1H identity ranking.
        # ==================================================

        if event_id in m1:

            m1_row = m1[event_id]

            rank = parse_rank(m1_row)

            # Verified reappearance existed for this event.
            row["target_returned"] = "yes"

            # Phase 1G retained the true target candidate
            # in all 10 eligible reappearance events.
            row[
                "target_detected_after_return"
            ] = "yes"

            # M1 identity ranking result.
            row[
                "target_selected_rank1"
            ] = (
                "yes"
                if rank == 1
                else "no"
            )

            # Phase 1H M1 memory-quality audit:
            # 26 recent memory entries checked,
            # 0 contaminated (< 0.30 tracker-vs-GT IoU).
            row[
                "memory_contaminated"
            ] = "no"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1G+Phase1H"
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                (
                    f"verified reappearance; "
                    f"target retained by Phase1G; "
                    f"M1 target rank={rank}; "
                    f"M1 memory audit found no contaminated "
                    f"recent-memory crops"
                )
            )

        # ==================================================
        # E012 — frozen no verified reappearance
        # ==================================================

        elif event_id == "E012":

            row["target_returned"] = "no"

            row[
                "target_detected_after_return"
            ] = "not_applicable"

            row[
                "target_selected_rank1"
            ] = "not_applicable"

            row[
                "memory_contaminated"
            ] = "unknown"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase1E frozen event review"
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                (
                    "no verified target reappearance "
                    "within the evaluated observation window"
                )
            )

        # ==================================================
        # E010 — deliberately unresolved
        # ==================================================

        elif event_id == "E010":

            row["target_returned"] = "unknown"

            row[
                "target_detected_after_return"
            ] = "unknown"

            row[
                "target_selected_rank1"
            ] = "unknown"

            row[
                "memory_contaminated"
            ] = "unknown"

            row["evidence_source"] = append_text(
                row["evidence_source"],
                "Phase0/Phase1 manual review"
            )

            row["evidence_notes"] = append_text(
                row["evidence_notes"],
                (
                    "target remained occluded; "
                    "not a verified target-absence case, "
                    "so return status is not forced"
                )
            )

    # --------------------------------------------------
    # Save updated taxonomy
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
    # Compact summary
    # --------------------------------------------------

    event_summary = {}

    for row in rows:

        event_id = row["event_id"]

        if event_id not in event_summary:

            event_summary[event_id] = {
                "returned":
                    row["target_returned"],

                "detected":
                    row[
                        "target_detected_after_return"
                    ],

                "rank1":
                    row[
                        "target_selected_rank1"
                    ],

                "memory":
                    row[
                        "memory_contaminated"
                    ],
            }

    for event_id in sorted(event_summary):

        x = event_summary[event_id]

        print(
            f"{event_id} | "
            f"returned={x['returned']:14} | "
            f"detected={x['detected']:14} | "
            f"rank1={x['rank1']:14} | "
            f"memory_contam={x['memory']}"
        )

    print()
    print("-" * 95)

    verified_return = sum(
        x["returned"] == "yes"
        for x in event_summary.values()
    )

    no_return = sum(
        x["returned"] == "no"
        for x in event_summary.values()
    )

    rank1_yes = sum(
        x["rank1"] == "yes"
        for x in event_summary.values()
    )

    rank1_no = sum(
        x["rank1"] == "no"
        for x in event_summary.values()
    )

    print(
        f"Verified-return events: "
        f"{verified_return}/12"
    )

    print(
        f"Verified no-return events: "
        f"{no_return}/12"
    )

    print(
        f"M1 Rank-1 selected correctly: "
        f"{rank1_yes}/10"
    )

    print(
        f"M1 Rank-1 selection failures: "
        f"{rank1_no}/10"
    )

    print()
    print(
        "Tracking-specific F1/F2/F3 facts remain untouched."
    )

    print(
        "F7 genuine ambiguity also remains unknown "
        "until explicit failure review."
    )

    print()
    print(f"Updated: {TAXONOMY}")


if __name__ == "__main__":
    main()
