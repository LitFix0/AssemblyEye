"""Model-assisted pre-labelling: build labelme DRAFTS that a human must review.

Drafts carry flags {"needs_review": true}. The YOLO exporter skips any file that still says true, so
unreviewed model output can never become a training or evaluation label.
"""
import copy
from pathlib import Path


def cap_to_expected(dets, category, required=None):
    """Keep the most confident detections up to the expected composition of a NORMAL image.
    Only valid for images known to be normal (the train/good folders)."""
    dets = sorted(dets, key=lambda d: -d.confidence)
    if category == "screw_bag":
        kept, seen = [], {}
        for d in dets:
            if seen.get(d.class_name, 0) < (required or {}).get(d.class_name, 0):
                kept.append(d)
                seen[d.class_name] = seen.get(d.class_name, 0) + 1
        return kept
    connectors = [d for d in dets if d.class_name.startswith("connector_")][:2]
    cables = [d for d in dets if d.class_name == "cable"][:1]
    return connectors + cables


def build_labelme(template: dict, image_name: str, width: int, height: int, dets) -> dict:
    """Copy the structure of a labelme file made by the real tool, so the format always matches."""
    data = copy.deepcopy(template)
    old = str(template.get("imagePath") or "")
    old_name = Path(old.replace("\\", "/")).name
    prefix = old[: len(old) - len(old_name)] if old_name else "../images/"
    base = copy.deepcopy(template["shapes"][0]) if template.get("shapes") else {}
    shapes = []
    for d in dets:
        s = copy.deepcopy(base)
        s.update({"label": d.class_name,
                  "points": [[d.bbox.x1, d.bbox.y1], [d.bbox.x2, d.bbox.y2]],
                  "group_id": None, "description": f"draft conf={d.confidence:.2f}",
                  "shape_type": "rectangle", "flags": {}})
        if "mask" in s:
            s["mask"] = None
        shapes.append(s)
    data.update({"shapes": shapes, "imagePath": prefix + image_name, "imageData": None,
                 "imageHeight": height, "imageWidth": width, "flags": {"needs_review": True}})
    return data