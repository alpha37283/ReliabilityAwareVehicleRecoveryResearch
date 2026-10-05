import csv
from pathlib import Path

from evaluation.event_loader import load_event


ROOT = Path(__file__).resolve().parents[2]

L2_FILE = ROOT / "results/phase1/phase1f_l2_results.csv"


def value_or_none(x):
    x = str(x).strip()

    if not x:
        return None

    return int(x)


def main():

    rows = []

    with L2_FILE.open(newline="") as f:
        for row in csv.DictReader(f):

            critical = value_or_none(
                row["persistent_critical_frame"]
            )

            lost = value_or_none(
                row["lost_frame"]
            )

            candidates = [
                x for x in [critical, lost]
                if x is not None
            ]

            if not candidates:
                continue

            trigger = min(candidates)

            event = load_event(
                row["event_id"]
            )

            reappearance = event.get(
                "reappearance_frame"
            )

            rows.append({
                "event_id": row["event_id"],
                "tracker": row["tracker"],
                "trigger_frame": trigger,
                "persistent_critical_frame":
                    critical,
                "lost_frame": lost,
                "reappearance_frame":
                    reappearance,
            })

    print("PHASE 1G — TRIGGERED RUNS")
    print("=" * 75)

    for row in rows:

        if row["reappearance_frame"] is not None:
            gap = (
                row["reappearance_frame"]
                - row["trigger_frame"]
            )
        else:
            gap = None

        print(
            f"{row['event_id']} | "
            f"{row['tracker']:7} | "
            f"trigger={row['trigger_frame']} | "
            f"critical={row['persistent_critical_frame']} | "
            f"lost={row['lost_frame']} | "
            f"reappearance={row['reappearance_frame']} | "
            f"gap={gap}"
        )

    print()
    print(f"Triggered tracker-event runs: {len(rows)}")


if __name__ == "__main__":
    main()
