# ST-03: Count-level results on the normal-only dev set

Date: 2026-10-28

Eval set: 60 normal images per category from `train/good` (the last 60 of the seeded shuffle, seed 42). Never annotated, never trained on.
This measures false rejects on NORMAL images only. It says nothing about missed defects.

Metric: share of images whose detections at confidence threshold t match the expected composition exactly.
- screw_bag: 1 screw_long, 1 screw_short, 2 nut, 2 washer, nothing else.
- splicing_connectors: 2 connectors of one type + 1 cable, nothing else.
- "suppressed": cross-class suppression (IoU 0.7, highest confidence wins) applied first.

## Runs (all: yolo11n, 100 epochs, batch 4, Colab T4, seed 42)

| Run | Train images | imgsz | Weights fingerprint | Git commit |
|---|---|---|---|---|
| smoke_screw_bag | 13 | 640 | 4a71791b786a | b5f8894 |
| screw_bag_v2 | 29 | 640 | 13f1b6d5333f | b869b1b |
| smoke_splicing | 13 | 960 | 197e11e04a99 | b5f8894 |
| splicing_v2 | 29 | 960 | 76c47f1e0822 | c08f4d0 |

## screw_bag: exact-count images out of 60 (raw / suppressed)

| thr | 13 images | 29 images |
|---|---|---|
| 0.3 | 2 / 12 (3% / 20%) | 44 / 47 (73% / 78%) |
| 0.4 | 2 / 12 | 45 / 45 (75%) |
| 0.5 | 2 / 7 | 41 / 42 (68% / 70%) |
| 0.6 | 0 / 0 | 37 / 37 (62%) |
| 0.7 | 0 / 0 | 30 / 30 (50%) |

Images with cross-class overlapping boxes at 0.3: 59 (13 images) vs 11 (29 images).

## splicing_connectors: exact-count images out of 60 (raw / suppressed)

| thr | 13 images | 29 images |
|---|---|---|
| 0.2 | 42 / 43 (70% / 72%) | 43 / 43 (72%) |
| 0.3 | 54 / 56 (90% / 93%) | 53 / 53 (88%) |
| 0.4 | 37 / 38 (62% / 63%) | 57 / 57 (95%) |
| 0.5 | 0 / 0 | 21 / 21 (35%) |
| 0.6, 0.7 | 0 / 0 | not run |

## Caveats

- One training run per setting, one seed. Run-to-run variance is unknown. 60 images means one image = 1.7 points, so differences of a few images (e.g. 90% vs 88% for splicing at 0.3) are not meaningful.
- Going from 13 to 29 images also doubled the optimizer steps (about 400 to about 800 at batch 4, 100 epochs). The gain is data plus steps, not data alone.
- The dev set has no defects. A threshold that is good here may still accept a hallucinated part where a defect should fail. No operating threshold is chosen until defect-bearing dev data is evaluated.
- Remaining screw_bag failures printed at 0.5 (5 of 19) are all screw errors (one screw missing or two of the same class); nuts and washers were 2/2 in all of them. The other 14 are not yet analysed.
