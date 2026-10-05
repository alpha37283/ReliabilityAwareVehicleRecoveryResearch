import csv
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]

METRICS = ROOT / "results/phase1/phase1e_final_metrics.csv"
CANDIDATES = ROOT / "results/phase1/phase1e_candidate_recall.csv"
OUTPUT = ROOT / "results/phase1/phase1e_summary.txt"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def b(value):
    return str(value).strip().lower() == "true"


def main():
    rows = read_csv(METRICS)
    candidates = read_csv(CANDIDATES)

    lines = []

    lines.append("PHASE 1E FINAL SUMMARY")
    lines.append("=" * 60)

    for tracker in ["sam21", "dam4sam"]:
        tr = [r for r in rows if r["tracker"] == tracker]

        continuous = sum(b(r["continuous_tracking"]) for r in tr)

        loss_runs = [
            r for r in tr
            if not b(r["continuous_tracking"])
        ]

        recovered = sum(
            r["automatic_recovery_status"]
            == "automatic_correct_recovery"
            for r in tr
        )

        no_verified_reappearance = sum(
            r["automatic_recovery_status"]
            == "loss_no_verified_reappearance"
            for r in tr
        )

        warnings = sum(
            int(r["switch_warning_count"])
            for r in tr
        )

        latency = mean(
            float(r["latency_ms"])
            for r in tr
        )

        fps = mean(
            float(r["fps"])
            for r in tr
        )

        lines.append("")
        lines.append(tracker.upper())
        lines.append("-" * 30)
        lines.append(f"Tracker-event runs: {len(tr)}")
        lines.append(f"Continuous-tracking runs: {continuous}")
        lines.append(f"Runs with automatic loss: {len(loss_runs)}")
        lines.append(f"Automatic correct recoveries: {recovered}")
        lines.append(
            f"Loss with no verified reappearance: "
            f"{no_verified_reappearance}"
        )
        lines.append(f"Total switch warnings: {warnings}")
        lines.append(f"Mean latency: {latency:.3f} ms/frame")
        lines.append(f"Mean FPS: {fps:.3f}")

    eligible = len(candidates)

    retained = sum(
        b(r["true_target_in_plausible"])
        for r in candidates
    )

    raw_total = sum(
        int(r["raw_candidate_count"])
        for r in candidates
    )

    plausible_total = sum(
        int(r["plausible_candidate_count"])
        for r in candidates
    )

    mean_candidates = mean(
        int(r["plausible_candidate_count"])
        for r in candidates
    )

    mean_distractors = mean(
        int(r["distractor_count"])
        for r in candidates
    )

    reduction = (
        1.0 - plausible_total / raw_total
        if raw_total else 0.0
    )

    lines.append("")
    lines.append("ANNOTATION-BACKED REFERENCE CANDIDATE GENERATION")
    lines.append("-" * 50)
    lines.append(f"Eligible events: {eligible}")
    lines.append(f"True target retained: {retained}/{eligible}")
    lines.append(
        f"Reference candidate retention: "
        f"{retained / eligible:.3f}"
    )
    lines.append(f"Raw candidates total: {raw_total}")
    lines.append(f"Plausible candidates total: {plausible_total}")
    lines.append(
        f"Candidate-set reduction: {reduction * 100:.1f}%"
    )
    lines.append(
        f"Mean plausible candidates/event: {mean_candidates:.2f}"
    )
    lines.append(
        f"Mean distractors/event: {mean_distractors:.2f}"
    )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 30)
    lines.append(
        "Candidate retention is annotation-backed reference retention, "
        "not detector candidate recall."
    )
    lines.append(
        "Automatic switch warnings are not confirmed identity switches."
    )
    lines.append(
        "Recovery is reported only after an automatically detected loss "
        "and not before the recorded reappearance frame."
    )
    lines.append(
        "E005-E007 reuse overlapping stationary-SUV footage and must not "
        "be treated as independent hard moving-target recovery examples."
    )

    text = "\n".join(lines)

    OUTPUT.write_text(text)
    print(text)
    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
