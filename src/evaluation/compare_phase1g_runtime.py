import csv
from pathlib import Path
from statistics import mean

from ultralytics import YOLO

from evaluation.event_loader import load_event


ROOT = Path(__file__).resolve().parents[2]

REFERENCE = ROOT / "results/phase1/phase1e_candidate_recall.csv"
IMAGE_ROOT = ROOT / "data/raw/UA-DETRAC/images"

SETTINGS = [
    (640, 0.20),
    (960, 0.25),
]


def main():
    model = YOLO("yolo11n.pt")

    with REFERENCE.open(newline="") as f:
        eligible = list(csv.DictReader(f))

    print("PHASE 1G — DETECTOR RUNTIME COMPARISON")
    print("=" * 65)

    for imgsz, conf in SETTINGS:

        inference_times = []
        preprocess_times = []
        postprocess_times = []

        for ref in eligible:

            event = load_event(ref["event_id"])

            sequence = event["sequence"]
            frame = int(event["reappearance_frame"])

            image_path = (
                IMAGE_ROOT
                / sequence
                / f"img{frame:05d}.jpg"
            )

            result = model.predict(
                source=str(image_path),
                imgsz=imgsz,
                conf=conf,
                device=0,
                classes=[2, 3, 5, 7],
                verbose=False,
            )[0]

            speed = result.speed

            preprocess_times.append(
                speed.get("preprocess", 0.0)
            )

            inference_times.append(
                speed.get("inference", 0.0)
            )

            postprocess_times.append(
                speed.get("postprocess", 0.0)
            )

        mean_pre = mean(preprocess_times)
        mean_inf = mean(inference_times)
        mean_post = mean(postprocess_times)

        total = (
            mean_pre
            + mean_inf
            + mean_post
        )

        fps = (
            1000.0 / total
            if total > 0
            else 0.0
        )

        print()
        print(
            f"imgsz={imgsz}, conf={conf}"
        )
        print(
            f"mean preprocess:  {mean_pre:.2f} ms"
        )
        print(
            f"mean inference:   {mean_inf:.2f} ms"
        )
        print(
            f"mean postprocess: {mean_post:.2f} ms"
        )
        print(
            f"mean total:       {total:.2f} ms"
        )
        print(
            f"approx FPS:       {fps:.2f}"
        )


if __name__ == "__main__":
    main()
