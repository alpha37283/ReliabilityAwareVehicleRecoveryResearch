import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]

QUALITY_FILE = (
    ROOT
    / "results/phase1/phase1h3_observation_quality.csv"
)

MARGIN_FILE = (
    ROOT
    / "results/phase1/phase1h1_candidate_margin_all.csv"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h3_runtime_routing_signals.csv"
)


def load_lookup(path):

    with path.open(newline="") as f:
        return {
            row["event_id"]: row
            for row in csv.DictReader(f)
        }


def main():

    quality = load_lookup(
        QUALITY_FILE
    )

    margin = load_lookup(
        MARGIN_FILE
    )

    rows = []

    print(
        "PHASE 1H-3 — RUNTIME ROUTING SIGNAL DIAGNOSTIC"
    )
    print("=" * 120)

    for event_id in quality:

        q = quality[event_id]
        m = margin[event_id]

        correct = (
            str(q["top1_correct"])
            .strip()
            .lower()
            == "true"
        )

        row = {
            "event_id":
                event_id,

            "top1_correct":
                correct,

            "top1_score":
                float(
                    q["top1_max_similarity"]
                ),

            "top1_top2_margin":
                float(
                    m["top1_top2_margin"]
                ),

            "top1_detector_confidence":
                float(
                    q["top1_detector_confidence"]
                ),

            "top1_area_ratio":
                float(
                    q["top1_area_ratio"]
                ),

            "top1_abs_log_area_change":
                float(
                    q["top1_abs_log_area_change"]
                ),

            "top1_aspect_ratio_ratio":
                float(
                    q["top1_aspect_ratio_ratio"]
                ),

            "top1_abs_log_aspect_change":
                float(
                    q["top1_abs_log_aspect_change"]
                ),

            "top1_mean_similarity":
                float(
                    q["top1_mean_similarity"]
                ),

            "top1_similarity_std":
                float(
                    q["top1_similarity_std"]
                ),
        }

        rows.append(row)

        print(
            f"{event_id} | "
            f"correct={str(correct):5} | "
            f"score={row['top1_score']:.4f} | "
            f"margin={row['top1_top2_margin']:+.4f} | "
            f"area={row['top1_area_ratio']:.2f}x | "
            f"|logA|={row['top1_abs_log_area_change']:.3f} | "
            f"conf={row['top1_detector_confidence']:.3f} | "
            f"AR={row['top1_aspect_ratio_ratio']:.2f}x"
        )

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

    correct_rows = [
        r for r in rows
        if r["top1_correct"]
    ]

    wrong_rows = [
        r for r in rows
        if not r["top1_correct"]
    ]

    print()
    print("-" * 120)

    metrics = [
        (
            "top1_score",
            "Top1 appearance score",
        ),
        (
            "top1_top2_margin",
            "Top1-Top2 margin",
        ),
        (
            "top1_abs_log_area_change",
            "Top1 |log area change|",
        ),
        (
            "top1_abs_log_aspect_change",
            "Top1 |log aspect change|",
        ),
        (
            "top1_detector_confidence",
            "Top1 detector confidence",
        ),
        (
            "top1_similarity_std",
            "Top1 memory similarity std",
        ),
    ]

    for key, label in metrics:

        c = [
            float(r[key])
            for r in correct_rows
        ]

        w = [
            float(r[key])
            for r in wrong_rows
        ]

        print(label)

        print(
            f"  correct mean = "
            f"{mean(c):.4f}"
        )

        print(
            f"  wrong mean   = "
            f"{mean(w):.4f}"
        )

        print(
            f"  correct range = "
            f"[{min(c):.4f}, {max(c):.4f}]"
        )

        print(
            f"  wrong range   = "
            f"[{min(w):.4f}, {max(w):.4f}]"
        )

        print()

    print(
        "No routing threshold is selected here."
    )

    print(
        "All diagnostic features above are "
        "runtime-available before GT labeling."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
