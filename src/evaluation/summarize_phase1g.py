import csv
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/phase1"

ORACLE_FILE = RESULTS / "phase1e_candidate_recall.csv"
REAL_FILTER_FILE = RESULTS / "phase1g_real_candidate_filtering.csv"
TRIGGER_FILE = RESULTS / "phase1g_triggered_candidate_results.csv"
L2_FILE = RESULTS / "phase1f_l2_results.csv"

OUTPUT_TXT = RESULTS / "phase1g_final_summary.txt"
OUTPUT_CSV = RESULTS / "phase1g_final_summary_metrics.csv"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_bool(x):
    return str(x).strip().lower() == "true"


def main():
    oracle = read_csv(ORACLE_FILE)
    real = read_csv(REAL_FILTER_FILE)
    triggered = read_csv(TRIGGER_FILE)
    l2 = read_csv(L2_FILE)

    # --------------------------------------------------
    # Oracle/reference candidate layer
    # --------------------------------------------------

    oracle_events = len(oracle)

    oracle_hits = sum(
        as_bool(r["true_target_in_plausible"])
        for r in oracle
    )

    oracle_raw = sum(
        int(r["raw_candidate_count"])
        for r in oracle
    )

    oracle_filtered = sum(
        int(r["plausible_candidate_count"])
        for r in oracle
    )

    oracle_reduction = (
        1.0 - oracle_filtered / oracle_raw
        if oracle_raw else 0.0
    )

    # --------------------------------------------------
    # REAL detector layer
    # --------------------------------------------------

    real_events = len(real)

    raw_real_hits = sum(
        as_bool(r["raw_target_retained"])
        for r in real
    )

    filtered_real_hits = sum(
        as_bool(r["filtered_target_retained"])
        for r in real
    )

    raw_real_total = sum(
        int(r["raw_candidate_count"])
        for r in real
    )

    filtered_real_total = sum(
        int(r["filtered_candidate_count"])
        for r in real
    )

    real_reduction = (
        1.0 - filtered_real_total / raw_real_total
        if raw_real_total else 0.0
    )

    # --------------------------------------------------
    # Trigger-driven evaluation
    # --------------------------------------------------

    eligible_triggered = [
        r for r in triggered
        if str(r["verified_reappearance_frame"]).strip()
    ]

    trigger_hits = sum(
        as_bool(r["target_proposed"])
        for r in eligible_triggered
    )

    delays = [
        int(r["candidate_proposal_delay_frames"])
        for r in eligible_triggered
        if str(r["candidate_proposal_delay_frames"]).strip()
    ]

    mean_delay = mean(delays) if delays else None

    total_detector_frames = sum(
        int(r["detector_frames_processed"])
        for r in triggered
    )

    # --------------------------------------------------
    # Genuine loss/reappearance cases E005-E007
    # --------------------------------------------------

    genuine_events = {"E005", "E006", "E007"}

    genuine_runs = [
        r for r in triggered
        if r["event_id"] in genuine_events
    ]

    genuine_hits = sum(
        as_bool(r["target_proposed"])
        for r in genuine_runs
    )

    # --------------------------------------------------
    # False Phase 1F triggers
    # --------------------------------------------------

    l2_lookup = {
        (r["event_id"], r["tracker"]): r
        for r in l2
    }

    false_trigger_runs = []

    for r in triggered:
        key = (r["event_id"], r["tracker"])

        phase1f = l2_lookup.get(key)

        if (
            phase1f is not None
            and phase1f["risk_classification"] == "FP"
        ):
            false_trigger_runs.append(r)

    false_trigger_frames = sum(
        int(r["detector_frames_processed"])
        for r in false_trigger_runs
    )

    # --------------------------------------------------
    # No-reappearance runs
    # --------------------------------------------------

    no_reappearance = [
        r for r in triggered
        if not str(
            r["verified_reappearance_frame"]
        ).strip()
    ]

    no_reappearance_frames = sum(
        int(r["detector_frames_processed"])
        for r in no_reappearance
    )

    # --------------------------------------------------
    # Text summary
    # --------------------------------------------------

    lines = []

    lines.append("PHASE 1G FINAL SUMMARY")
    lines.append("=" * 65)

    lines.append("")
    lines.append("FROZEN REAL-MODE DETECTOR")
    lines.append("-" * 40)
    lines.append("Model: YOLO11n")
    lines.append("Input size: 640")
    lines.append("Confidence threshold: 0.20")
    lines.append("Classes: car, motorcycle, bus, truck")
    lines.append("Candidate evaluation IoU threshold: 0.30")
    lines.append(
        "Plausibility filter: area ratio 0.35-15.0, "
        "aspect-ratio ratio 0.5-2.0"
    )

    lines.append("")
    lines.append("ORACLE / DEBUG MODE")
    lines.append("-" * 40)
    lines.append(
        f"Eligible events: {oracle_events}"
    )
    lines.append(
        f"True target retained: "
        f"{oracle_hits}/{oracle_events}"
    )
    lines.append(
        f"Reference retention: "
        f"{oracle_hits/oracle_events:.3f}"
    )
    lines.append(
        f"Candidates: "
        f"{oracle_raw} -> {oracle_filtered}"
    )
    lines.append(
        f"Candidate reduction: "
        f"{oracle_reduction*100:.1f}%"
    )

    lines.append("")
    lines.append("REAL MODE — RECORDED REAPPEARANCE FRAMES")
    lines.append("-" * 50)
    lines.append(
        f"Eligible events: {real_events}"
    )
    lines.append(
        f"Raw detector target recall: "
        f"{raw_real_hits}/{real_events} "
        f"= {raw_real_hits/real_events:.3f}"
    )
    lines.append(
        f"Filtered target recall: "
        f"{filtered_real_hits}/{real_events} "
        f"= {filtered_real_hits/real_events:.3f}"
    )
    lines.append(
        f"Candidates: "
        f"{raw_real_total} -> {filtered_real_total}"
    )
    lines.append(
        f"Candidate reduction: "
        f"{real_reduction*100:.1f}%"
    )
    lines.append(
        f"Mean filtered candidates/event: "
        f"{filtered_real_total/real_events:.2f}"
    )

    lines.append("")
    lines.append("TRIGGER-DRIVEN REAL MODE")
    lines.append("-" * 40)
    lines.append(
        f"Total Phase 1F-triggered runs: "
        f"{len(triggered)}"
    )
    lines.append(
        f"Triggered runs with verified reappearance: "
        f"{len(eligible_triggered)}"
    )
    lines.append(
        f"Target eventually proposed: "
        f"{trigger_hits}/{len(eligible_triggered)}"
    )

    if mean_delay is not None:
        lines.append(
            f"Mean measured proposal delay: "
            f"{mean_delay:.2f} frames"
        )

    lines.append(
        f"Total detector frames processed after triggers: "
        f"{total_detector_frames}"
    )

    lines.append("")
    lines.append("GENUINE LOSS / REAPPEARANCE CASES")
    lines.append("-" * 40)
    lines.append(
        f"E005-E007 tracker-runs: "
        f"{len(genuine_runs)}"
    )
    lines.append(
        f"Successful target proposals: "
        f"{genuine_hits}/{len(genuine_runs)}"
    )

    lines.append("")
    lines.append("FALSE-TRIGGER COST")
    lines.append("-" * 40)
    lines.append(
        f"False Phase 1F risk triggers: "
        f"{len(false_trigger_runs)}"
    )
    lines.append(
        f"Detector frames caused by false triggers: "
        f"{false_trigger_frames}"
    )

    lines.append("")
    lines.append("NO-REAPPEARANCE CASES")
    lines.append("-" * 40)
    lines.append(
        f"Runs without verified reappearance: "
        f"{len(no_reappearance)}"
    )
    lines.append(
        f"Detector frames processed in these runs: "
        f"{no_reappearance_frames}"
    )
    lines.append(
        "These runs are not counted as successful or failed "
        "candidate recovery because no verified target "
        "reappearance exists."
    )

    lines.append("")
    lines.append("INTERPRETATION")
    lines.append("-" * 40)
    lines.append(
        "1. Oracle/debug candidate generation and real detector "
        "candidate generation are kept separate."
    )
    lines.append(
        "2. The frozen YOLO11n real detector retained the target "
        "in all 10 tested recorded-reappearance events."
    )
    lines.append(
        "3. Geometric plausibility filtering reduced the real "
        "candidate set without removing the target in the tested set."
    )
    lines.append(
        "4. In genuine triggered loss/reappearance cases E005-E007, "
        "the target was available in the candidate set on the first "
        "verified reappearance frame."
    )
    lines.append(
        "5. A zero-frame measured proposal delay means the detector "
        "already contained the target when recovery evaluation became "
        "valid; it does not prove independent visibility onset detection."
    )
    lines.append(
        "6. E004 demonstrates the computational consequence of a "
        "false Phase 1F reliability trigger."
    )
    lines.append(
        "7. Phase 1G proposes plausible vehicles only; it does not "
        "identify which candidate is the selected target."
    )

    text = "\n".join(lines)

    OUTPUT_TXT.write_text(text)

    summary_row = {
        "oracle_events": oracle_events,
        "oracle_target_hits": oracle_hits,
        "oracle_candidate_reduction":
            round(oracle_reduction, 4),

        "real_events": real_events,
        "real_raw_target_hits": raw_real_hits,
        "real_filtered_target_hits":
            filtered_real_hits,

        "real_raw_candidates":
            raw_real_total,

        "real_filtered_candidates":
            filtered_real_total,

        "real_candidate_reduction":
            round(real_reduction, 4),

        "triggered_runs":
            len(triggered),

        "triggered_eligible_reappearance_runs":
            len(eligible_triggered),

        "triggered_target_hits":
            trigger_hits,

        "mean_candidate_proposal_delay_frames":
            mean_delay,

        "genuine_loss_reappearance_runs":
            len(genuine_runs),

        "genuine_target_hits":
            genuine_hits,

        "false_trigger_runs":
            len(false_trigger_runs),

        "false_trigger_detector_frames":
            false_trigger_frames,

        "no_reappearance_runs":
            len(no_reappearance),

        "total_detector_frames":
            total_detector_frames,
    }

    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_row.keys()),
        )
        writer.writeheader()
        writer.writerow(summary_row)

    print(text)
    print()
    print(f"Saved: {OUTPUT_TXT}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
