import csv
from pathlib import Path

from ultralytics import YOLO

from evaluation.event_loader import load_event
from recovery.annotation_candidate_generator import get_frame_candidates
from recovery.real_candidate_generator import filter_real_candidates


ROOT = Path(__file__).resolve().parents[2]

L2_FILE = ROOT / "results/phase1/phase1f_l2_results.csv"

OUTPUT = (
    ROOT
    / "results/phase1/phase1g_triggered_candidate_results.csv"
)

IMAGE_ROOT = ROOT / "data/raw/UA-DETRAC/images"

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

IMGSZ = 640
CONF = 0.20
MATCH_IOU = 0.30

POST_REAPPEARANCE_WINDOW = 30


def value_or_none(x):
    x = str(x).strip()
    return int(x) if x else None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    aa = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
    bb = max(0.0, bx2-bx1) * max(0.0, by2-by1)

    union = aa + bb - inter

    return inter / union if union > 0 else 0.0


def get_triggered_runs():
    runs = []

    with L2_FILE.open(newline="") as f:
        for row in csv.DictReader(f):

            critical = value_or_none(
                row["persistent_critical_frame"]
            )

            lost = value_or_none(
                row["lost_frame"]
            )

            valid = [
                x for x in (critical, lost)
                if x is not None
            ]

            if not valid:
                continue

            runs.append({
                "event_id": row["event_id"],
                "tracker": row["tracker"],
                "trigger_frame": min(valid),
            })

    return runs


def get_prediction_end(event_id, tracker):
    path = (
        ROOT
        / "results/phase1"
        / event_id
        / tracker
        / "predictions.csv"
    )

    last = None

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            last = int(row["frame_number"])

    return last


def main():

    model = YOLO("yolo11n.pt")

    triggered = get_triggered_runs()

    rows = []

    print("PHASE 1G — TRIGGER-DRIVEN REAL CANDIDATE GENERATION")
    print("=" * 90)

    for run in triggered:

        event_id = run["event_id"]
        tracker = run["tracker"]
        trigger = run["trigger_frame"]

        event = load_event(event_id)

        sequence = event["sequence"]
        target_id = int(event["target_id"])
        reappearance = event.get("reappearance_frame")

        selection_box = event["selection_box"]

        if isinstance(selection_box, dict):
            reference_box = selection_box["xyxy"]
        else:
            reference_box = selection_box

        xml_path = XML_ROOT / f"{sequence}.xml"

        prediction_end = get_prediction_end(
            event_id,
            tracker,
        )

        if reappearance is not None:
            search_end = min(
                prediction_end,
                reappearance + POST_REAPPEARANCE_WINDOW,
            )
        else:
            search_end = prediction_end

        detector_frames = 0
        pre_reappearance_frames = 0

        first_target_frame = None
        first_target_iou = None

        raw_at_first = None
        filtered_at_first = None

        for frame in range(trigger, search_end + 1):

            image_path = (
                IMAGE_ROOT
                / sequence
                / f"img{frame:05d}.jpg"
            )

            if not image_path.exists():
                continue

            result = model.predict(
                source=str(image_path),
                imgsz=IMGSZ,
                conf=CONF,
                device=0,
                classes=[2, 3, 5, 7],
                verbose=False,
            )[0]

            detector_frames += 1

            if (
                reappearance is not None
                and frame < reappearance
            ):
                pre_reappearance_frames += 1

            raw = []

            for box in result.boxes:

                cls_id = int(box.cls[0])

                raw.append({
                    "box_xyxy":
                        box.xyxy[0].tolist(),

                    "confidence":
                        float(box.conf[0]),

                    "class_id":
                        cls_id,

                    "class_name":
                        result.names[cls_id],
                })

            filtered = filter_real_candidates(
                raw,
                reference_box,
            )

            # --------------------------------------------------
            # Do NOT evaluate recovery before verified
            # reappearance.
            # --------------------------------------------------

            if reappearance is None:
                continue

            if frame < reappearance:
                continue

            gt = get_frame_candidates(
                str(xml_path),
                frame,
            )

            target = next(
                (
                    g for g in gt
                    if int(g["target_id"]) == target_id
                ),
                None,
            )

            if target is None:
                continue

            target_box = target["box_xyxy"]

            best = max(
                (
                    iou(
                        target_box,
                        candidate["box_xyxy"],
                    )
                    for candidate in filtered
                ),
                default=0.0,
            )

            if (
                best >= MATCH_IOU
                and first_target_frame is None
            ):
                first_target_frame = frame
                first_target_iou = best

                raw_at_first = len(raw)
                filtered_at_first = len(filtered)

                # Once target is proposed, candidate
                # generation has achieved its job for
                # this event.
                break

        if (
            reappearance is not None
            and first_target_frame is not None
        ):
            proposal_delay = (
                first_target_frame
                - reappearance
            )

            target_proposed = True

        else:
            proposal_delay = None
            target_proposed = False

        rows.append({
            "event_id": event_id,
            "tracker": tracker,

            "trigger_frame":
                trigger,

            "verified_reappearance_frame":
                reappearance,

            "detector_frames_processed":
                detector_frames,

            "pre_reappearance_detector_frames":
                pre_reappearance_frames,

            "first_target_candidate_frame":
                first_target_frame,

            "candidate_proposal_delay_frames":
                proposal_delay,

            "target_proposed":
                target_proposed,

            "target_iou_at_first_proposal":
                (
                    round(first_target_iou, 4)
                    if first_target_iou is not None
                    else ""
                ),

            "raw_candidates_at_first_proposal":
                (
                    raw_at_first
                    if raw_at_first is not None
                    else ""
                ),

            "filtered_candidates_at_first_proposal":
                (
                    filtered_at_first
                    if filtered_at_first is not None
                    else ""
                ),
        })

        print(
            f"{event_id} | "
            f"{tracker:7} | "
            f"trigger={trigger} | "
            f"reappear={reappearance} | "
            f"first_candidate={first_target_frame} | "
            f"delay={proposal_delay} | "
            f"processed={detector_frames}"
        )

    with OUTPUT.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    eligible = [
        row for row in rows
        if row["verified_reappearance_frame"] is not None
    ]

    hits = sum(
        row["target_proposed"]
        for row in eligible
    )

    delays = [
        row["candidate_proposal_delay_frames"]
        for row in eligible
        if row["candidate_proposal_delay_frames"]
        is not None
    ]

    print()
    print("-" * 90)

    print(
        f"Eligible triggered runs with verified reappearance: "
        f"{len(eligible)}"
    )

    print(
        f"Target eventually proposed: "
        f"{hits}/{len(eligible)}"
    )

    if delays:
        print(
            f"Mean candidate proposal delay: "
            f"{sum(delays)/len(delays):.2f} frames"
        )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
