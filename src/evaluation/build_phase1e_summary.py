import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

AUTO_FILE = ROOT / "results/phase1/phase1e_event_metrics.csv"
MANUAL_FILE = ROOT / "results/phase1/phase1e_manual_outcome_audit.csv"

OUTPUT_FILE = ROOT / "results/phase1/phase1e_final_metrics.csv"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main():
    automatic = read_csv(AUTO_FILE)
    manual = read_csv(MANUAL_FILE)

    manual_lookup = {
        (row["event_id"], row["tracker"]): row
        for row in manual
    }

    final_rows = []

    for auto in automatic:
        key = (auto["event_id"], auto["tracker"])

        if key not in manual_lookup:
            raise ValueError(
                f"Missing manual audit for {key[0]}/{key[1]}"
            )

        man = manual_lookup[key]

        row = {
            "event_id": auto["event_id"],
            "tracker": auto["tracker"],

            "pre_occlusion_tracking":
                auto["pre_occlusion_tracking"],

            "pre_occlusion_tracking_rate":
                auto["pre_occlusion_tracking_rate"],

            "continuous_tracking":
                auto["continuous_tracking"],

            "loss_frame":
                auto["loss_frame"],

            "recorded_reappearance_frame":
                auto["recorded_reappearance_frame"],

            "automatic_recovery_status":
                auto["recovery_status"],

            "recovery_frame":
                auto["recovery_frame"],

            "recovery_delay_frames":
                auto["recovery_delay_frames"],

            "switch_warning_count":
                auto["switch_warning_count"],

            "first_switch_warning_frame":
                auto["first_switch_warning_frame"],

            "confirmed_id_switch":
                man["confirmed_id_switch"],

            "false_recovery":
                man["false_recovery"],

            "manual_tracking_outcome":
                man["manual_tracking_outcome"],

            "candidate_retained":
                auto["candidate_retained"],

            "candidate_count":
                auto["candidate_count"],

            "distractor_count":
                auto["distractor_count"],

            "latency_ms":
                auto["latency_ms"],

            "fps":
                auto["fps"],

            "manual_notes":
                man["manual_notes"],
        }

        final_rows.append(row)

    fields = [
        "event_id",
        "tracker",

        "pre_occlusion_tracking",
        "pre_occlusion_tracking_rate",

        "continuous_tracking",

        "loss_frame",
        "recorded_reappearance_frame",

        "automatic_recovery_status",
        "recovery_frame",
        "recovery_delay_frames",

        "switch_warning_count",
        "first_switch_warning_frame",
        "confirmed_id_switch",
        "false_recovery",

        "manual_tracking_outcome",

        "candidate_retained",
        "candidate_count",
        "distractor_count",

        "latency_ms",
        "fps",

        "manual_notes",
    ]

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(final_rows)}")

    print()
    print("Important runs:")
    print("-" * 80)

    for row in final_rows:
        if (
            row["continuous_tracking"] == "False"
            or row["switch_warning_count"] != "0"
        ):
            print(
                f"{row['event_id']} | "
                f"{row['tracker']:7} | "
                f"loss={row['loss_frame'] or 'None'} | "
                f"recovery={row['recovery_frame'] or 'None'} | "
                f"warnings={row['switch_warning_count']} | "
                f"manual={row['manual_tracking_outcome']}"
            )


if __name__ == "__main__":
    main()
