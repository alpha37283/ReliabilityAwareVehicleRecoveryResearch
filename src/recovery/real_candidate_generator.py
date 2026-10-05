def box_properties(box):
    x1, y1, x2, y2 = box

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    area = width * height

    aspect = (
        width / height
        if height > 0
        else 0.0
    )

    return {
        "width": width,
        "height": height,
        "area": area,
        "aspect": aspect,
    }


def filter_real_candidates(
    detections,
    reference_box,
    min_area_ratio=0.35,
    max_area_ratio=15.0,
    min_aspect_ratio=0.5,
    max_aspect_ratio=2.0,
):
    """
    Filter detector proposals using only geometric
    plausibility relative to the user-selected vehicle.

    No ground-truth identity information is used.

    Expected detection format:
        {
            "box_xyxy": [...],
            "confidence": float,
            "class_id": int,
            "class_name": str,
        }
    """

    ref = box_properties(reference_box)

    if ref["area"] <= 0 or ref["aspect"] <= 0:
        raise ValueError("Invalid reference box.")

    kept = []

    for det in detections:

        props = box_properties(
            det["box_xyxy"]
        )

        if props["area"] <= 0:
            continue

        area_ratio = (
            props["area"] / ref["area"]
        )

        aspect_ratio = (
            props["aspect"] / ref["aspect"]
            if ref["aspect"] > 0
            else 0.0
        )

        if not (
            min_area_ratio
            <= area_ratio
            <= max_area_ratio
        ):
            continue

        if not (
            min_aspect_ratio
            <= aspect_ratio
            <= max_aspect_ratio
        ):
            continue

        out = dict(det)

        out["area_ratio"] = area_ratio
        out["aspect_ratio"] = aspect_ratio

        kept.append(out)

    return kept
