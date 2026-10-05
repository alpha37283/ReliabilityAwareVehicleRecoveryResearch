import csv
import math
from pathlib import Path
from statistics import mean

import torch
from PIL import Image

from evaluation.event_loader import load_event
from recovery.appearance_encoder import AppearanceEncoder
from recovery.annotation_candidate_generator import get_frame_candidates


ROOT = Path(__file__).resolve().parents[2]

ELIGIBLE = (
    ROOT
    / "results/phase1/phase1e_candidate_recall.csv"
)

IMAGE_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/images"
)

XML_ROOT = (
    ROOT
    / "data/raw/UA-DETRAC/annotations/test_xml"
    / "DETRAC-Test-Annotations-XML"
)

OUTPUT = (
    ROOT
    / "results/phase1/phase1h1_m2_distractor_opportunities.csv"
)

# Diagnostic sampling only.
FRAME_STEP = 5

# Not yet frozen as M2 thresholds.
# These are descriptive flags for the audit.
SIMILARITY_FLAG = 0.80
NORMALIZED_DISTANCE_FLAG = 3.0


def clamp_box(box, width, height):

    x1, y1, x2, y2 = box

    return [
        int(max(0, min(width, round(float(x1))))),
        int(max(0, min(height, round(float(y1))))),
        int(max(0, min(width, round(float(x2))))),
        int(max(0, min(height, round(float(y2))))),
    ]


def center(box):

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
    )


def diagonal(box):

    x1, y1, x2, y2 = box

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    return math.sqrt(
        w * w + h * h
    )


def normalized_center_distance(
    target_box,
    distractor_box,
):

    tx, ty = center(target_box)
    dx, dy = center(distractor_box)

    distance = math.sqrt(
        (tx - dx) ** 2
        + (ty - dy) ** 2
    )

    return (
        distance
        / diagonal(target_box)
    )


def crop_box(image, box):

    box = clamp_box(
        box,
        image.width,
        image.height,
    )

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop(box)


def main():

    encoder = AppearanceEncoder()

    with ELIGIBLE.open(newline="") as f:
        events = list(csv.DictReader(f))

    rows = []

    print(
        "PHASE 1H-1 — M2 DISTRACTOR-OPPORTUNITY AUDIT"
    )
    print("=" * 108)

    for item in events:

        event_id = item["event_id"]
        event = load_event(event_id)

        sequence = event["sequence"]
        target_id = int(event["target_id"])

        selection_frame = int(
            event["selection_frame"]
        )

        occlusion_start = int(
            event["occlusion_start"]
        )

        xml_path = (
            XML_ROOT
            / f"{sequence}.xml"
        )

        # Sample pre-occlusion interval.
        frames = list(
            range(
                selection_frame,
                occlusion_start,
                FRAME_STEP,
            )
        )

        # Always include final visible frame immediately
        # before annotated occlusion start.
        final_pre_occ = occlusion_start - 1

        if (
            final_pre_occ >= selection_frame
            and final_pre_occ not in frames
        ):
            frames.append(final_pre_occ)

        frames = sorted(set(frames))

        event_rows = []

        for frame in frames:

            image_path = (
                IMAGE_ROOT
                / sequence
                / f"img{frame:05d}.jpg"
            )

            if not image_path.exists():
                continue

            objects = get_frame_candidates(
                str(xml_path),
                frame,
            )

            target = next(
                (
                    obj
                    for obj in objects
                    if int(obj["target_id"])
                    == target_id
                ),
                None,
            )

            if target is None:
                continue

            image = Image.open(
                image_path
            ).convert("RGB")

            target_box = target["box_xyxy"]

            target_crop = crop_box(
                image,
                target_box,
            )

            if target_crop is None:
                continue

            target_embedding = (
                encoder.encode(
                    target_crop
                )
            )

            distractors = [
                obj
                for obj in objects
                if int(obj["target_id"])
                != target_id
            ]

            best = None

            for distractor in distractors:

                distractor_box = (
                    distractor["box_xyxy"]
                )

                distractor_crop = crop_box(
                    image,
                    distractor_box,
                )

                if distractor_crop is None:
                    continue

                distractor_embedding = (
                    encoder.encode(
                        distractor_crop
                    )
                )

                similarity = float(
                    torch.dot(
                        target_embedding,
                        distractor_embedding,
                    ).item()
                )

                norm_distance = (
                    normalized_center_distance(
                        target_box,
                        distractor_box,
                    )
                )

                candidate = {
                    "event_id":
                        event_id,

                    "frame_number":
                        frame,

                    "target_id":
                        target_id,

                    "distractor_id":
                        int(
                            distractor[
                                "target_id"
                            ]
                        ),

                    "similarity":
                        similarity,

                    "normalized_center_distance":
                        norm_distance,

                    "similarity_flag":
                        similarity
                        >= SIMILARITY_FLAG,

                    "nearby_flag":
                        norm_distance
                        <= NORMALIZED_DISTANCE_FLAG,

                    "hard_distractor_flag":
                        (
                            similarity
                            >= SIMILARITY_FLAG
                            and
                            norm_distance
                            <= NORMALIZED_DISTANCE_FLAG
                        ),
                }

                if (
                    best is None
                    or
                    candidate["similarity"]
                    > best["similarity"]
                ):
                    best = candidate

            if best is not None:

                rows.append(best)
                event_rows.append(best)

        hard = [
            r
            for r in event_rows
            if r["hard_distractor_flag"]
        ]

        similar = [
            r
            for r in event_rows
            if r["similarity_flag"]
        ]

        if event_rows:

            strongest = max(
                event_rows,
                key=lambda x:
                    x["similarity"],
            )

            print(
                f"{event_id} | "
                f"sampled={len(event_rows):2d} | "
                f"similar_frames={len(similar):2d} | "
                f"hard_frames={len(hard):2d} | "
                f"best_sim="
                f"{strongest['similarity']:.4f} | "
                f"best_frame="
                f"{strongest['frame_number']} | "
                f"distractor="
                f"{strongest['distractor_id']} | "
                f"norm_dist="
                f"{strongest['normalized_center_distance']:.2f}"
            )

        else:

            print(
                f"{event_id} | "
                "no usable distractor observations"
            )

    with OUTPUT.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "frame_number",
                "target_id",
                "distractor_id",
                "similarity",
                "normalized_center_distance",
                "similarity_flag",
                "nearby_flag",
                "hard_distractor_flag",
            ],
        )

        writer.writeheader()

        for row in rows:

            out = row.copy()

            out["similarity"] = round(
                out["similarity"],
                6,
            )

            out[
                "normalized_center_distance"
            ] = round(
                out[
                    "normalized_center_distance"
                ],
                4,
            )

            writer.writerow(out)

    print()
    print("-" * 108)

    if rows:

        events_with_hard = {
            row["event_id"]
            for row in rows
            if row[
                "hard_distractor_flag"
            ]
        }

        events_with_similar = {
            row["event_id"]
            for row in rows
            if row[
                "similarity_flag"
            ]
        }

        print(
            f"Events with visually similar "
            f"pre-occlusion distractor: "
            f"{len(events_with_similar)}/"
            f"{len(events)}"
        )

        print(
            f"Events with similar + nearby "
            f"hard distractor: "
            f"{len(events_with_hard)}/"
            f"{len(events)}"
        )

        print(
            f"Mean strongest-frame similarity: "
            f"{mean([r['similarity'] for r in rows]):.4f}"
        )

    print()
    print(
        "NOTE: similarity/distance thresholds above "
        "are diagnostic flags only, not frozen M2 rules."
    )

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
