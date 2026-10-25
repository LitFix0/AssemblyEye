# ST-00: Dataset Audit and Initial Technical Plan

Date: 2026-10-24
Dataset: MVTec LOCO AD, categories `screw_bag` and `splicing_connectors`
Licence: CC BY-NC-SA 4.0 (non-commercial, ShareAlike). Cite Bergmann et al., "Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization", IJCV 2022.

Evidence: output of `scripts/audit_dataset.py`, `scripts/inspect_dataset_samples.py`, and visual review of `scripts/make_labeled_samples.py` contact sheets.
Counts below are measured. Items marked (hypothesis) are interpretations still to be verified.

## 1. Contents (measured)

| | screw_bag | splicing_connectors |
|---|---|---|
| train/good | 360 | 360 |
| validation/good | 60 | 60 |
| test/good | 122 | 119 |
| test/logical_anomalies | 137 | 108 |
| test/structural_anomalies | 82 | 85 |
| Image size (RGB PNG) | 1600x1100 | 1700x850 |

- Train and validation contain only normal images. All defective images are in test/.
- Ground truth: one 8-bit grayscale PNG mask per defect per test image, same size as the image.
- Screw_bag logical has 137 images but 161 masks: 24 images carry two defects. The same image number appears under two defect types in the contact sheet. Findings must therefore support multiple defects per image.

## 2. Annotations

- Image-level labels come from folder names. Defect type comes from the mask pixel value, mapped by `defects_config.json`.
- There are NO bounding boxes and NO component labels. Anomaly masks are not component boxes.
- YOLO cannot be trained from the provided annotations. We must create component boxes ourselves. Because annotations of this data are adaptations, release them under CC BY-NC-SA if shared (not legal advice).

## 3. Defect catalog (pixel value: name)

screw_bag logical (16): 255 screw_too_long, 254 screw_too_short, 253 1_very_short_screw, 252 2_very_short_screws, 251 1_additional_long_screw, 250 1_additional_short_screw, 249 1_additional_nut, 248 2_additional_nuts, 247 1_additional_washer, 246 2_additional_washers, 245 1_missing_long_screw, 244 1_missing_short_screw, 243 1_missing_nut, 242 2_missing_nuts, 241 1_missing_washer, 240 2_missing_washers.
screw_bag structural (4): 239 bag_broken, 238 color, 237 contamination, 236 part_broken.

splicing logical (13 observed): 255/254/253 wrong_connector_type_5_2 / 5_3 / 3_2, 252/251/250 cable_too_short_T2 / T3 / T5, 249 missing_connector, 248 missing_connector_and_cable, 247 missing_cable, 246 extra_cable, 245 cable_color, 243 cable_cut, 240 wrong_cable_location.
splicing structural (8 observed): 244 broken_cable, 242 cable_not_plugged, 241 unknown_cable_color, 239 flipped_connector, 238 broken_connector, 237 open_lever, 236 color, 235 contamination.

## 4. What the masks mark (from visual review)

- Localized: length defects mark the screw segment; single additional nut/washer/screw mark the items; structural defects mark small regions.
- Whole-bag tint (screw_bag): all `missing_*` types, `2_additional_nuts`, `2_additional_washers`. The anomaly is a property of the whole bag, not a location.
- Whole-image tint (splicing): `cable_too_short_T2/T3/T5` and `missing_connector_and_cable`. This matches the measured 14 logical masks with no background pixels (12 + 2).
- Splicing `missing_connector` and `missing_cable` mark the region where the part should be.
- Consequence: these types cannot be used to score localization. Evaluate them at image level only and report them separately.

## 5. Specification types differ per category

- screw_bag is a bill of materials with counts and size classes. Parts lie loose in a bag, so there are no slots. (hypothesis) Normal = 1 long screw, 1 short screw, 2 nuts, 2 washers, inferred from the defect names and 8 viewed normal images.
- splicing_connectors is relational: two connectors of the same clamp count (2, 3 or 5) joined by one cable. (hypothesis) Cable color follows connector type (yellow/2, blue/3, red/5), seen in a few images only.
- The verification engine must therefore support both count-based and relation-based rules.

## 6. Technical plan

1. Detector classes (default, to confirm in ST-02): screw_bag = screw_long, screw_short, nut, washer. splicing = connector_2, connector_3, connector_5, cable. Cable color is computed from pixels in verification (decision to confirm in ST-02).
2. Every component class appears in normal images, and defects arise from combinations. Annotate about 100 normal images per category from train/good (plus part of validation/good as a held-out detector check). Keep the whole test set sealed and unannotated for training.
3. Expectation for Phase 1 (hypothesis to test in ST-08/ST-10): count, type and presence defects are catchable by detection plus rules. Most structural defects (contamination, tears, color, small breaks) probably are not, and belong to the Phase 2 anomaly branch.
4. Splits: train/good for training, validation/good for threshold tuning, test/ untouched until final evaluation.
5. Compute: RTX 3050 Laptop (4.3 GB). Local for development, Colab for YOLO and PatchCore training.

## 7. ST-01 (planned)

Pipeline skeleton with hand-written detection fixtures for 3 images (normal, missing part, extra part): input validation, schemas, a simple count rule, visualization, JSON result. Labelled as a demonstration, not a detector.

## 8. Open items

- Review the splicing_connectors normal sheets (not yet reviewed).
- Verify the normal composition by counting during annotation (including whether split washers appear in normal images).
- Decide annotation tool in ST-02, before labelling starts.
