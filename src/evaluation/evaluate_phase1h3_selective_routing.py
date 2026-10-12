import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MARGIN_FILE = (
    ROOT
    / "results/phase1/phase1h1_candidate_margin_all.csv"
)

CONFIDENCE_FILE = (
    ROOT
    / "results/phase1/phase1h3_confidence_consistency.csv"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h3_selective_routing.csv"
)


# ----------------------------------------------------------
# EXPLORATORY PILOT THRESHOLDS ONLY.
# These are NOT final/frozen research thresholds.
# ----------------------------------------------------------

MIN_APPEARANCE_MARGIN = 0.03
MIN_CONFIDENCE_PERCENTILE = 0.50


def load_lookup(path):

    with path.open(newline="") as f:
        return {
            row["event_id"]: row
            for row in csv.DictReader(f)
        }


def parse_bool(value):

    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


def main():

    margins = load_lookup(
        MARGIN_FILE
    )

    confidences = load_lookup(
        CONFIDENCE_FILE
    )

    rows = []

    print(
        "PHASE 1H-3 — R1 VS EXPLORATORY R2 SELECTIVE ROUTING"
    )
    print("=" * 115)

    for event_id in margins:

        margin_row = margins[event_id]
        confidence_row = confidences[event_id]

        correct = parse_bool(
            margin_row["top1_correct"]
        )

        appearance_margin = float(
            margin_row[
                "top1_top2_margin"
            ]
        )

        confidence_percentile = float(
            confidence_row[
                "top1_confidence_percentile"
            ]
        )

        detector_confidence = float(
            confidence_row[
                "top1_detector_confidence"
            ]
        )

        appearance_score = float(
            confidence_row[
                "top1_appearance_score"
            ]
        )

        # --------------------------------------------------
        # R1
        # Always accept Rank-1.
        # --------------------------------------------------

        r1_decision = "MATCH"

        # --------------------------------------------------
        # R2
        # Selective routing using two independent
        # runtime-visible reliability signals.
        # --------------------------------------------------

        margin_reliable = (
            appearance_margin
            >= MIN_APPEARANCE_MARGIN
        )

        observation_reliable = (
            confidence_percentile
            >= MIN_CONFIDENCE_PERCENTILE
        )

        if (
            margin_reliable
            and observation_reliable
        ):
            r2_decision = "MATCH"
        else:
            r2_decision = "NEED_MORE_EVIDENCE"

        r2_match_correct = (
            r2_decision == "MATCH"
            and correct
        )

        r2_false_match = (
            r2_decision == "MATCH"
            and not correct
        )

        # Why was R2 cautious?
        reasons = []

        if not margin_reliable:
            reasons.append(
                "low_candidate_margin"
            )

        if not observation_reliable:
            reasons.append(
                "low_observation_reliability"
            )

        reason = (
            "+".join(reasons)
            if reasons
            else "accepted"
        )

        rows.append({
            "event_id":
                event_id,

            "top1_correct":
                correct,

            "appearance_score":
                round(
                    appearance_score,
                    6,
                ),

            "appearance_margin":
                round(
                    appearance_margin,
                    6,
                ),

            "detector_confidence":
                round(
                    detector_confidence,
                    6,
                ),

            "confidence_percentile":
                round(
                    confidence_percentile,
                    6,
                ),

            "r1_decision":
                r1_decision,

            "r2_decision":
                r2_decision,

            "r2_reason":
                reason,

            "r2_match_correct":
                r2_match_correct,

            "r2_false_match":
                r2_false_match,
        })

        print(
            f"{event_id} | "
            f"GT={'CORRECT' if correct else 'WRONG':7} | "
            f"margin={appearance_margin:.4f} | "
            f"conf_pct={confidence_percentile:.3f} | "
            f"R1=MATCH | "
            f"R2={r2_decision:18} | "
            f"{reason}"
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # R1 summary
    # --------------------------------------------------

    n = len(rows)

    r1_accepted = n

    r1_correct = sum(
        row["top1_correct"]
        for row in rows
    )

    r1_false = (
        r1_accepted
        - r1_correct
    )

    r1_accuracy = (
        r1_correct
        / r1_accepted
    )

    # --------------------------------------------------
    # R2 summary
    # --------------------------------------------------

    r2_matches = [
        row
        for row in rows
        if row["r2_decision"] == "MATCH"
    ]

    r2_need_more = [
        row
        for row in rows
        if row["r2_decision"]
        == "NEED_MORE_EVIDENCE"
    ]

    r2_accepted = len(
        r2_matches
    )

    r2_correct = sum(
        row["r2_match_correct"]
        for row in rows
    )

    r2_false = sum(
        row["r2_false_match"]
        for row in rows
    )

    coverage = (
        r2_accepted / n
    )

    selective_accuracy = (
        r2_correct
        / r2_accepted
        if r2_accepted > 0
        else 0.0
    )

    need_more_rate = (
        len(r2_need_more)
        / n
    )

    false_match_rate_all = (
        r2_false / n
    )

    print()
    print("-" * 115)

    print(
        "R1 — ALWAYS ACCEPT APPEARANCE RANK-1"
    )

    print(
        f"Accepted MATCH decisions: "
        f"{r1_accepted}/{n}"
    )

    print(
        f"Correct accepted matches: "
        f"{r1_correct}/{r1_accepted}"
    )

    print(
        f"False accepted matches: "
        f"{r1_false}/{r1_accepted}"
    )

    print(
        f"Accepted-match accuracy: "
        f"{r1_accuracy:.3f}"
    )

    print()

    print(
        "R2 — SELECTIVE ROUTING"
    )

    print(
        f"Accepted MATCH decisions: "
        f"{r2_accepted}/{n}"
    )

    print(
        f"Correct accepted matches: "
        f"{r2_correct}/{r2_accepted}"
        if r2_accepted
        else
        "Correct accepted matches: 0/0"
    )

    print(
        f"False accepted matches: "
        f"{r2_false}/{r2_accepted}"
        if r2_accepted
        else
        "False accepted matches: 0/0"
    )

    print(
        f"Selective accuracy: "
        f"{selective_accuracy:.3f}"
    )

    print(
        f"Coverage: "
        f"{coverage:.3f}"
    )

    print(
        f"NEED_MORE_EVIDENCE: "
        f"{len(r2_need_more)}/{n} "
        f"= {need_more_rate:.3f}"
    )

    print(
        f"False MATCH rate over all events: "
        f"{r2_false}/{n} "
        f"= {false_match_rate_all:.3f}"
    )

    print()

    print(
        "Exploratory thresholds:"
    )

    print(
        f"  appearance margin >= "
        f"{MIN_APPEARANCE_MARGIN:.2f}"
    )

    print(
        f"  confidence percentile >= "
        f"{MIN_CONFIDENCE_PERCENTILE:.2f}"
    )

    print()

    print(
        "These thresholds are pilot-derived and "
        "must NOT be presented as validated final thresholds."
    )

    print()

    print(
        f"Saved: {OUTPUT}"
    )


if __name__ == "__main__":
    main()
