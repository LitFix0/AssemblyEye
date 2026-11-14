# Models

Detectors for the Visual Assembly Verification System (research prototype, not validated on production data).
Weights are NOT committed (see .gitignore). This file records how each carried-forward detector was made.

## Carried-forward detectors

| | screw_bag_A_s42 | splicing_v2 |
|---|---|---|
| Classes | screw_long, screw_short, nut, washer | connector_2, connector_3, connector_5, cable |
| Base model | yolo11n.pt (COCO-pretrained, Ultralytics) | same |
| Images used (train / val / holdout) | 29 / 7 / 8 | 29 / 7 / 8 |
| Training | 100 epochs, batch 4, imgsz 640, seed 42, Colab T4 | 100 epochs, batch 4, imgsz 960, seed 42, Colab T4 |
| Weights fingerprint (first 12 hex of SHA-256) | 5ac8ab93847f | 76c47f1e0822 |
| Run records (git commit, library versions, dataset report) | reports/training/screw_bag_A_s42/provenance.json | reports/training/splicing_v2/provenance.json |

All training images are normal images from the dataset's train/good and validation/good folders. No defective image has been used for training.

## Measured results

Box-level, 8-image holdout (normal images, never trained on, never used to choose settings):

| | boxes | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| screw_bag: screw_long | 8 | 0.993 | 1.000 | 0.995 | 0.874 |
| screw_bag: screw_short | 8 | 0.978 | 1.000 | 0.995 | 0.782 |
| screw_bag: nut | 16 | 0.988 | 1.000 | 0.995 | 0.873 |
| screw_bag: washer | 16 | 0.989 | 1.000 | 0.995 | 0.911 |
| screw_bag: all | 48 | 0.987 | 1.000 | 0.995 | 0.860 |
| splicing: connector_2 | 8 | 0.934 | 1.000 | 0.995 | 0.766 |
| splicing: connector_5 | 8 | 0.925 | 1.000 | 0.995 | 0.771 |
| splicing: cable | 8 | 0.796 | 1.000 | 0.967 | 0.675 |
| splicing: all | 24 | 0.885 | 1.000 | 0.986 | 0.737 |

Count-level, 60 normal dev images (share with exactly the expected parts): screw_bag_A_s42 reaches 53-57 of 60 at confidence thresholds 0.3-0.7; splicing_v2 reaches 53 of 60 at 0.3, 57 at 0.4 and 21 at 0.5. See reports/st03_results.md.

Verdict-level (screw_bag only, DEV split of the test set, placeholder thresholds accept 0.6 / review 0.3): 3 of 37 normals flagged, 42 of 43 logical defects flagged, 10 of 26 structural defects flagged. Splicing has no verdict-level result yet (needs the ST-04 relational rules).

## Limitations

- Tiny data: 29 training images per detector. Box-level metrics come from 8 holdout images and are anecdotal; they cannot rank models.
- The holdout contains only normal images, so near-perfect box metrics say nothing about defects.
- screw_bag: long vs short is decided by the detector from appearance. It is the main source of false alarms and also what flags the screw length defects. The system measures no length.
- splicing: connector_3 is absent from the holdout. Cable confidence typically sits around 0.4-0.5, so a single global threshold is fragile.
- Structural defects (contamination, color, bag_broken, part_broken) are mostly not caught for the right reason; flags on them are partly confidence noise.
- Most settings were run once or twice. Seed-to-seed spread on the 60 dev images was 2-4 images for the two-class screw detector.
- Thresholds are untuned placeholders. The FINAL part of the test split has not been touched.
- Clean lab images only. Not validated on production data and not factory-ready.

## Licences (not legal advice)

- Ultralytics code and pretrained weights: AGPL-3.0 (enterprise licence available). Weights fine-tuned from them are derivatives.
- MVTec LOCO AD: CC BY-NC-SA 4.0, non-commercial; cite Bergmann et al., IJCV 2022. Models trained on it are for non-commercial research use unless licensing is resolved.
