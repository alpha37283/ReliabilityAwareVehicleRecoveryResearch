import csv
import time
from pathlib import Path
from statistics import mean, median, stdev

import torch
from ultralytics import YOLO

from evaluation.event_loader import load_event


ROOT = Path(__file__).resolve().parents[2]

REFERENCE = ROOT / "results/phase1/phase1e_candidate_recall.csv"
IMAGE_ROOT = ROOT / "data/raw/UA-DETRAC/images"

SETTINGS = [
    (640, 0.20),
    (960, 0.25),
]

REPEATS = 5


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    model = YOLO("yolo11n.pt")

    with REFERENCE.open(newline="") as f:
        eligible = list(csv.DictReader(f))

    image_paths = []

    for ref in eligible:
        event = load_event(ref["event_id"])

        image_paths.append(
            IMAGE_ROOT
            / event["sequence"]
            / f"img{int(event['reappearance_frame']):05d}.jpg"
        )

    # CUDA/model warm-up
    print("Warming up...")

    for _ in range(3):
        model.predict(
            source=str(image_paths[0]),
            imgsz=640,
            conf=0.20,
            device=0,
            classes=[2, 3, 5, 7],
            verbose=False,
        )

    sync()

    print("\nPHASE 1G — ROBUST END-TO-END BENCHMARK")
    print("=" * 65)

    for imgsz, conf in SETTINGS:

        times = []

        for repeat in range(REPEATS):

            for image_path in image_paths:

                sync()
                start = time.perf_counter()

                model.predict(
                    source=str(image_path),
                    imgsz=imgsz,
                    conf=conf,
                    device=0,
                    classes=[2, 3, 5, 7],
                    verbose=False,
                )

                sync()

                elapsed_ms = (
                    time.perf_counter() - start
                ) * 1000.0

                times.append(elapsed_ms)

        avg = mean(times)
        med = median(times)
        sd = stdev(times)

        print()
        print(f"imgsz={imgsz}, conf={conf}")
        print(f"Measurements: {len(times)}")
        print(f"Mean latency:   {avg:.2f} ms")
        print(f"Median latency: {med:.2f} ms")
        print(f"Std deviation:  {sd:.2f} ms")
        print(f"Approx FPS:     {1000.0 / avg:.2f}")


if __name__ == "__main__":
    main()
