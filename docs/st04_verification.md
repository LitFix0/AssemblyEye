# ST-04: Deterministic verification engine

## Goal
Turn detections into a PASS / REVIEW / FAIL decision using explicit rules, with findings that carry
evidence and operator-readable text. No model and no LLM inside the decision: same detections + spec +
image give the same result. Uncertainty goes to REVIEW, never to PASS.

## Data flow
detector -> `list[Detection]` -> `verify_product(detections, spec, image_bgr)` -> `(Status, list[Finding])`

`engine.verify_product` picks the rule set from the spec:
- spec without `splicing` -> `count_rules.verify_counts` (screw_bag; never reads pixels)
- spec with `splicing` -> `splicing.verify_splicing` (needs the image: cable colour comes from pixels)

`pipeline.run_inspection` now calls `verify_product`. `Finding` gained an optional `evidence` dict.

## Confidence bands
Per class (a spec may give `class_thresholds`): `>= accept_conf` confident, `[review_conf, accept_conf)`
uncertain, below that ignored as noise. Splicing gives the cable its own thresholds (accept 0.40 /
review 0.20) because its confidence sits around 0.4-0.5.

## Rules and finding codes
| Code | Status | Raised when |
|---|---|---|
| MISSING / EXTRA | FAIL | confident count below / above the spec (existing ST-01 rule, unchanged) |
| UNCERTAIN_PART / UNCERTAIN_EXTRA | REVIEW | count is right only if low-confidence detections are included / one too many appears |
| UNEXPECTED_CLASS | REVIEW | confident detection of a class the spec does not know |
| DUPLICATE_DETECTION | REVIEW | same-class boxes with IoU >= `duplicate_iou` (0.7): counted once |
A below-accept box that overlaps a stronger same-class box at IoU >= `weak_duplicate_iou` (0.4, splicing only)
is dropped silently: it is the same object detected again. A confident second box is never dropped this way.
| COMPETING_CLASSES | REVIEW | one object read as two classes of a competing group (IoU >= `competing_iou`): more confident reading kept |
| CLASS_SPLIT | REVIEW | a competing group (long/short screw) has the right total but wrong split, or too few confident parts but enough with low-confidence ones |
| CONNECTOR_TYPE_MISMATCH | FAIL | the two connectors are different types |
| CABLE_NOT_CONNECTED | FAIL | gap between cable box and a connector box > `end_tolerance` x image diagonal |
| CABLE_MISPLACED | FAIL | cable sticks out past the far side of a connector by more than the tolerance |
| CABLE_TOO_SHORT | FAIL | cable length along the connector axis < `min_cable_length` x image diagonal |
| CABLE_COLOR_MISMATCH | FAIL if the wrong colour holds >= 70% of coloured pixels, else REVIEW | colour read from pixels differs from the one the connector type needs |
| CABLE_COLOR_UNKNOWN | REVIEW | too few coloured pixels, or no colour clearly wins |

Splicing order: duplicates -> competing connector classes -> connector count (2) and cable count (1) ->
connector types equal -> cable ends -> cable colour. Geometry and colour run only when there are exactly
two confident connectors and one confident cable; otherwise the count findings already say what is wrong.
The axis between the connectors is detected, so left/right and upper/lower layouts both work.

## Decisions you should know about
1. **Long/short screw confusion is REVIEW, not FAIL** (your handoff rule). On the saved screw_bag DEV
   report this can move length-defect images from FAIL to REVIEW. They are still flagged; only the
   label changes. Revisit the policy in ST-10 when FINAL is used.
2. **Duplicates** are off in the engine by default (`duplicate_iou: null`) and on in both shipped specs.
   The detector's own NMS already merges same-class boxes at IoU 0.7, so on real detections this is mostly a
   safety net.
3. **Free-position products** (screw_bag) have no "misplaced" rule: only counts matter. Position rules
   exist only for splicing.
4. **Mixed connector types** (confident) are FAIL. connector_3 is absent from the 8-image holdout, so this
   class is the least tested.

## Untuned placeholders (all in `configs/products/splicing_connectors.yaml`)
- cable thresholds 0.40 / 0.20: from the ST-03 count sweep on the 60 normal dev images (57/60 exact at 0.4).
- `end_tolerance` 0.02 (about 38 px at 1700x850): on the 44 annotated normals the ground-truth cable end is
  at worst about 3 px short of its connector. Detector boxes are noisier than that.
- Hue ranges (yellow 20-35, blue 100-130, red 0-8 and 172-179), saturation/brightness floors, pixel and
  share limits: guesses. No splicing image was available while writing this. Run
  `scripts/check_splicing_rules.py` before trusting a cable-colour FAIL.
- screw_bag thresholds stay 0.6 / 0.3 as before.
- `weak_duplicate_iou` 0.4: on the DEV normals the weak second cable box overlaps the real one at IoU
  0.50-0.70; the one real second cable I measured overlaps at 0.00-0.29. Chosen from DEV, based on few examples.
- `min_cable_length` 0.18 x diagonal (about 342 px): normal cables are 591-1001 px (DEV) and 654-962 px
  (44 annotated normals). Derived from normals only; the three DEV too-short cables (162-172 px) were not used.

## Limitations
- Tested on synthetic pixels and real-geometry boxes; not yet on real detector output for splicing.
- Cable colour assumes a neutral, unsaturated background and that connector pixels are removed by their
  boxes. Lighting changes will move hues.
- "Cable cut", "cable not plugged" and similar defects are only caught if they change the cable box
  geometry or colour. Structural defects remain mostly invisible to a box-based check (ST-09 hypothesis).
- Nearest-end geometry uses axis-aligned boxes: a diagonal cable has a larger box than its true extent.
- Measured on DEV only (FINAL sealed until ST-10); thresholds were chosen from DEV, so treat as tuned.
- splicing DEV (98): normals 34 PASS / 2 REVIEW / 0 FAIL; logical 8 PASS / 3 REVIEW / 24 FAIL;
  structural 21 PASS / 2 REVIEW / 4 FAIL. `wrong_cable_location` mostly passes (6 of 8): no rule covers it yet.
  The 2 normal REVIEWs are cables at 0.38-0.39, just under the 0.40 accept threshold.
- screw_bag DEV (106): normals 34 PASS / 3 REVIEW / 0 FAIL (with cross-class suppression; 32 / 5 / 0 without).
  All 24 count defects FAIL; 17 length defects are REVIEW by design; 1 `1_very_short_screw` passes.
- The weak-duplicate rule could hide a real second cable lying on top of the first.

## Commands (PowerShell, from the repo root)
```powershell
cd "$env:USERPROFILE\Desktop\Visual Assembly Verification System"
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```
EXPECTED: all tests pass (142 in the sandbox where this was written: the original 78 plus 64 new).

```powershell
python scripts\check_splicing_rules.py
```
EXPECTED: a table with one row per annotated image (44 in the committed labels). Every row should be PASS
with the colour matching the connector type. Rows that are not PASS are listed with their messages at the end.

```powershell
$w = Get-ChildItem -Path . -Recurse -Filter splicing_v2_best.pt -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\.venv\\' } | Select-Object -First 1
python scripts\eval_defects.py --category splicing_connectors --weights $w.FullName --imgsz 960
```
EXPECTED: a table like the screw_bag one for the splicing DEV split (98 images: 36 normal, 35 logical, 27
structural) and a list of why normals were flagged. Paste it back before drawing conclusions.

```powershell
$w = Get-ChildItem -Path . -Recurse -Filter screw_bag_A_s42_best.pt -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\.venv\\' } | Select-Object -First 1
python scripts\eval_defects.py --category screw_bag --weights $w.FullName --imgsz 640 --accept 0.6 --review 0.3
python scripts\eval_defects.py --category screw_bag --weights $w.FullName --imgsz 640 --accept 0.6 --review 0.3 --no-suppress
```
EXPECTED: normals unchanged or nearly; some logical FAILs (length defects) now REVIEW. The second command
lets the engine handle one-object-two-classes itself instead of the external suppression.

## Common errors
- `ModuleNotFoundError: app...` when running a script: run from the repo root, scripts add `backend` to the path.
- `splicing verification needs the image`: call `verify_product(dets, spec, img.bgr)`, not `verify_counts`.
- `spec ... does not match category`: pass the matching `--spec` or omit it.
- Cable colour UNKNOWN on many normals: the hue ranges or saturation floor do not fit your images. Check the
  `check_splicing_rules.py` output first.

## Acceptance checklist
- [x] Six required cases covered by unit tests for screw_bag and for splicing (correct, missing, misplaced,
      duplicate, wrong class, uncertain) plus count, competition, colour and spec-validation tests
- [x] Result is deterministic and independent of detection order (tested)
- [x] Nothing uncertain can PASS (tested)
- [x] Evidence and plain-language text on every new finding
- [x] Existing tests still pass (one fixture adjusted: `normal_parts` boxes spaced apart)
- [x] `check_splicing_rules.py` run on the 44 annotated normals: 44/44 PASS
- [x] First splicing DEV evaluation run and reviewed (DEV only; FINAL stays sealed)
- [x] screw_bag DEV re-run reviewed (competition rule: length defects now REVIEW, not FAIL)
- [ ] Loose end: confirm last commit/push; screenshots of the contamination images if you want them looked at
