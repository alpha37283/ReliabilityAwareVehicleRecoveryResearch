import xml.etree.ElementTree as ET
from pathlib import Path

def filter_plausible_candidates(
    candidates,
    reference_box_xyxy,
    min_area_ratio=0.35,
    max_area_ratio=3.0,
    min_aspect_ratio=0.5,
    max_aspect_ratio=2.0,
):
    """
    Cheap high-recall candidate gate.

    Uses only size/shape consistency with the user-selected target.
    It does NOT use candidate target_id for filtering.
    """

    x1, y1, x2, y2 = reference_box_xyxy

    ref_width = x2 - x1
    ref_height = y2 - y1
    ref_area = ref_width * ref_height
    ref_aspect = ref_width / max(ref_height, 1e-6)

    plausible = []

    for candidate in candidates:
        area_ratio = candidate["area"] / max(ref_area, 1e-6)

        aspect = (
            candidate["width"] /
            max(candidate["height"], 1e-6)
        )

        aspect_ratio = aspect / max(ref_aspect, 1e-6)

        if (
            min_area_ratio <= area_ratio <= max_area_ratio
            and
            min_aspect_ratio <= aspect_ratio <= max_aspect_ratio
        ):
            candidate = dict(candidate)
            candidate["area_ratio"] = area_ratio
            candidate["aspect_ratio_ratio"] = aspect_ratio
            plausible.append(candidate)

    return plausible


def get_frame_candidates(xml_path, frame_number):
    """
    Return every UA-DETRAC vehicle annotation present in one frame.

    This is an annotation-backed REFERENCE candidate source.
    It is not a deployable detector and must not be reported as detector recall.
    """

    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    root = ET.parse(xml_path).getroot()

    wanted_frame = None

    for frame in root.findall(".//frame"):
        if int(frame.get("num")) == int(frame_number):
            wanted_frame = frame
            break

    if wanted_frame is None:
        return []

    candidates = []

    for target in wanted_frame.findall("./target_list/target"):

        target_id = int(target.get("id"))

        box = target.find("box")
        if box is None:
            continue

        attribute = target.find("attribute")

        vehicle_type = None
        if attribute is not None:
            vehicle_type = attribute.get("vehicle_type")

        left = float(box.get("left"))
        top = float(box.get("top"))
        width = float(box.get("width"))
        height = float(box.get("height"))

        candidates.append({
            "target_id": target_id,
            "vehicle_type": vehicle_type,
            "box_xyxy": [
                left,
                top,
                left + width,
                top + height,
            ],
            "width": width,
            "height": height,
            "area": width * height,
        })

    return candidates


if __name__ == "__main__":
    import argparse

# argument section 

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--target-id", type=int, default=None)
    parser.add_argument(
    "--reference-box",
    nargs=4,
    type=float,
    metavar=("X1", "Y1", "X2", "Y2"),
    help="Target reference box used for the plausibility gate",)

    args = parser.parse_args()

    candidates = get_frame_candidates(args.xml, args.frame)

    print(f"Frame {args.frame}")
    print(f"Candidates: {len(candidates)}")

    found_target = False

    for i, candidate in enumerate(candidates, start=1):
        marker = ""

        if (
            args.target_id is not None
            and candidate["target_id"] == args.target_id
        ):
            marker = "  <-- TRUE TARGET"
            found_target = True

        print(
            f"{i:02d}. "
            f"ID={candidate['target_id']} "
            f"type={candidate['vehicle_type']} "
            f"box={candidate['box_xyxy']}"
            f"{marker}"
        )

    if args.target_id is not None:
        print()
        print(f"True target present in reference candidates: {found_target}")

    candidates = get_frame_candidates(args.xml, args.frame)

    if args.reference_box is not None:
        plausible = filter_plausible_candidates(
            candidates,
            args.reference_box
        )
    else:
        plausible = candidates

    print(f"All candidates: {len(candidates)}")
    print(f"Plausible candidates: {len(plausible)}")

    target_found = False

    for i, candidate in enumerate(plausible, start=1):
        marker = ""

        if (
            args.target_id is not None
            and candidate["target_id"] == args.target_id
        ):
            marker = "  <-- TRUE TARGET"
            target_found = True

        print(
            f"{i:02d}. "
            f"ID={candidate['target_id']} "
            f"type={candidate['vehicle_type']} "
            f"area_ratio={candidate['area_ratio']:.2f} "
            f"aspect_ratio={candidate['aspect_ratio_ratio']:.2f}"
            f"{marker}"
        )

    print()
    print(
        f"True target retained after plausibility gate: "
        f"{target_found}"
    )
