# ST-03: Results log

Date: 2026-11-14

## Evaluation sets

- Normal dev: 60 normal images per category from train/good (last 60 of the seeded shuffle, seed 42). Never annotated, never trained on. Measures false rejects only.
- Test split: official test images split once, stratified by defect type, into DEV (about 30%) and FINAL (sealed, not touched). Frozen in data/splits/. screw_bag DEV = 106 images (37 normal, 43 logical, 26 structural).
- Holdout: 8 annotated normal images per category from validation/good. Box-level metrics only.

Count-level metric: share of images whose detections at confidence threshold t match the expected composition exactly (screw_bag: 1 long, 1 short, 2 nuts, 2 washers; splicing: 2 connectors of one type + 1 cable).

## Training runs (all yolo11n, 100 epochs, batch 4, Colab T4)

| Run | Train images | imgsz | Weights fingerprint |
|---|---|---|---|
| smoke_screw_bag | 13 | 640 | 4a71791b786a |
| screw_bag_v2 (train_074 labels swapped) | 29 | 640 | 13f1b6d5333f |
| screw_bag_A_s42 / A_s7 (labels corrected) | 29 | 640 | 5ac8ab93847f / fef7d4239b6f |
| screw_bag_B_s42 / B_s7 (one merged screw class + length rule) | 29 | 640 | 5c4ad9197ea9 / 551039351734 |
| smoke_splicing | 13 | 960 | 197e11e04a99 |
| splicing_v2 | 29 | 960 | 76c47f1e0822 |

## Count-level results, 60 normal dev images (exact-composition images, raw)

| thr | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| screw_bag smoke (13 images) | 2 | 2 | 2 | 0 | 0 |
| screw_bag v2 | 44 | 45 | 41 | 37 | 30 |
| screw_bag A seed 42 | 56 | 57 | 56 | 55 | 53 |
| screw_bag A seed 7 | 53 | 54 | 54 | 53 | 49 |
| screw_bag B seed 42 | 56 | 56 | 57 | 58 | 57 |
| screw_bag B seed 7 | 47 | 50 | 50 | 52 | 51 |

| thr | 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|
| splicing smoke (13 images) | 42 | 54 | 37 | 0 |
| splicing v2 | 43 | 53 | 57 | 21 |

Long/short screw design: B (merged class + length rule) had to beat A at 0.3, 0.4 and 0.5 for both seeds. It did not (seed 7: 47/50/50 vs 53/54/54). Decision: keep two classes (A) and carry forward seed 42, the default seed used throughout. The seed spread inside A (2-4 images) is about as large as the A-vs-B differences, so the experiment cannot separate the designs.

## Verdict-level results, screw_bag DEV split (placeholder thresholds accept 0.6, review 0.3, cross-class suppression on)

| | n | PASS | REVIEW | FAIL |
|---|---|---|---|---|
| normal, A seed 42 | 37 | 34 | 1 | 2 |
| logical, A seed 42 | 43 | 1 | 0 | 42 |
| structural, A seed 42 | 26 | 16 | 5 | 5 |
| normal, A seed 7 | 37 | 32 | 2 | 3 |
| logical, A seed 7 | 43 | 0 | 1 | 42 |
| structural, A seed 7 | 26 | 13 | 6 | 7 |

Threshold sweep (accept 0.5/0.6/0.7 x review 0.2/0.3/0.4, with and without suppression, seed 42): logical defects flagged 42/43 in all 18 settings. Normals flagged 3-7 of 37, structural 8-18 of 26. Accept 0.7 gave 7/37 normals and 18/26 structural flagged in every setting; accept 0.5-0.6 gave 3-5/37 and 8-11/26. The review threshold barely mattered. Dropping suppression changed hard FAILs on normals by one image either way and flagged 1-2 more normals overall. Differences of 1-2 images on 37 are within noise. Placeholder thresholds kept.

## Why defects were flagged (finding classes, seed 42, 0.6/0.3)

- screw_too_long: 6/6 flagged as EXTRA screw_long + MISSING screw_short (both screws read as long). screw_too_short: 6/6 flagged as MISSING screw_long + EXTRA screw_short. 1_very_short_screw 4/5 and 2_very_short_screws 2/2 show the same pattern. The detector's long/short class tracks screw length; the system measures no length, so a length defect that keeps the class pair intact passes (one DEV example).
- The 3 flagged normals are all screw problems (long/short class near the boundary).
- Of 10 flagged structural images, about 5 have findings that plausibly concern the defect (extra nut/washer on an odd or foreign object) and 5 are screw-confidence noise. Not yet confirmed on the images.

## Holdout box-level metrics (8 images each, normal only)

screw_bag_A_s42: precision 0.987, recall 1.000, mAP50 0.995, mAP50-95 0.860 (48 boxes).
splicing_v2: precision 0.885, recall 1.000, mAP50 0.986, mAP50-95 0.737 (24 boxes; no connector_3 in the holdout).
Per-class values are in models/README.md.

## Caveats

- 60 / 37 / 8 images per measure; one image is 1.7 / 2.7 / 12 points. Most settings are a single run or two seeds.
- Normal-image measures say nothing about defects; verdict-level is screw_bag only; splicing needs the ST-04 rules.
- High box-level metrics on the holdout coexist with 3-12% false alarms at verdict level, because mAP ignores the fixed thresholds the rules use.
- FINAL split untouched.
