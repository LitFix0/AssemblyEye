# ST-02: Data Preparation

Date: 2026-10-27
Scope: screw_bag and splicing_connectors (MVTec LOCO AD, CC BY-NC-SA 4.0, non-commercial)

## Pipeline

raw dataset -> adapter (`backend/app/datasets/loco.py`) -> seeded selection (`selection.py`, seed 42)
-> annotation workspace (`prepare.py`, `data/annotation/<category>/`) -> labelme
-> `scripts/export_yolo_dataset.py` -> `data/processed/yolo_<category>/` (YOLO txt labels + data.yaml)
-> `scripts/check_yolo_labels.py` (round-trip images and box-size report)

## Selection and leakage guard

- Only normal images are annotated. `train_pool` comes from `train/good`, `holdout` from `validation/good`. Nothing from `test/` is ever selected.
- The guard is enforced three times: at selection, at manifest check (prepare and export), and in unit tests.
- Selections are prefixes of one seeded shuffle, so a batch can be extended without changing existing labels. Shrinking a batch is refused.
- Current batch: 16 train_pool + 4 holdout per category.

## Splits

- `train_pool` is split 80/20 into train/val by seeded hash. `holdout` is kept apart and exported as `test:` in data.yaml. It is for final detector metrics only.
- Current size per category: train 13 / val 3 / holdout 4.

## Classes (configs/datasets/loco_classes.yaml)

- screw_bag: screw_long, screw_short, nut, washer
- splicing_connectors: connector_2, connector_3, connector_5, cable (number = orange clamp count)
- Cable color is not a detector class. It is to be computed from pixels in the verification stage.

## Annotation

- Tool: labelme 7.9.1 (own venv). Guidelines: docs/annotation_guidelines.md.
- Rectangles, polygons and circles are accepted and converted to tight axis-aligned bounding boxes.
- screw_bag: 120 boxes (80 polygon, 40 circle). splicing_connectors: 60 boxes (all polygon).
- The exporter rejects unknown classes, other shape types, boxes outside the image, boxes under 2 px, and size mismatches, and reports all problems at once.

## Measured results

| | screw_bag | splicing_connectors |
|---|---|---|
| Images (train/val/holdout) | 13 / 3 / 4 | 13 / 3 / 4 |
| Boxes | screw_long 20, screw_short 20, nut 40, washer 40 | connector_2 14, connector_3 10, connector_5 16, cable 20 |
| Composition check | 20/20 images = 1 long screw, 1 short screw, 2 nuts, 2 washers | 20/20 images = 2 connectors of one type + 1 cable |

Smallest box side after resizing to imgsz 640: nut 45 px, washer 45 px, screw_short 46 px, screw_long 47 px; cable 18 px (median about 30 px), connector_2 75 px, connector_3 105 px, connector_5 107 px.

screw_bag round-trip image reviewed by eye (8 of 20 images): boxes sit on the correct parts.

## Known limitations

- Splits are far too small for meaningful metrics. The first training run is a pipeline smoke test, not a performance claim.
- splicing class balance: connector_3 has 3 training images and none in the holdout; connector_5 has none in val. Per-class numbers are not interpretable yet.
- Axis-aligned boxes on diagonal screws are loose and overlap neighbouring parts. Long vs short screw must be separated by visible length. Watch for confusion in ST-03.
- The composition check is a sample of 20 images per category. The screw_bag spec stays a placeholder until verified on more data.
- The cable-color rule (yellow/2, blue/3, red/5) was seen in 8 of 8 viewed images and cannot be checked from these labels.
- The cable is thin: its smallest side is about 18-21 px at imgsz 640, so splicing likely needs a larger imgsz (to be measured in ST-03).

## Licence

Annotations derive from CC BY-NC-SA 4.0 images. Cite Bergmann et al., IJCV 2022. Annotations shared publicly should carry the same licence. Images themselves are not committed.
