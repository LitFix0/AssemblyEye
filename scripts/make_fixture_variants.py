import json
from pathlib import Path

base = Path("data/fixtures")
normal = json.loads((base / "screw_bag_normal.json").read_text(encoding="utf-8"))["detections"]


def write(name, dets):
    (base / f"screw_bag_{name}.json").write_text(json.dumps({"detections": dets}, indent=2), encoding="utf-8")
    print("wrote", name, f"({len(dets)} detections)")


# missing nut: drop the first nut
dets = [dict(d) for d in normal]
i = next(k for k, d in enumerate(dets) if d["class_name"] == "nut")
write("missing_nut", dets[:i] + dets[i + 1:])

# low-confidence nut: first nut at 0.45
dets = [dict(d) for d in normal]
i = next(k for k, d in enumerate(dets) if d["class_name"] == "nut")
dets[i]["confidence"] = 0.45
write("low_conf_nut", dets)

# extra washer: duplicate the first washer
dets = [dict(d) for d in normal]
i = next(k for k, d in enumerate(dets) if d["class_name"] == "washer")
write("extra_washer", dets + [dict(dets[i])])