# yolo26n on HSLVision

yolo26n object detector for the RoboCup Humanoid Soccer League, trained on
[whirlwind-ams/hslvision](https://huggingface.co/datasets/whirlwind-ams/hslvision)
and exported to ONNX for deployment.

**Trained model:** `runs/yolo26n-hslvision/weights/best.onnx` (10.1 MB, opset 12)

## Results — held-out test split (911 images, 9,841 boxes)

| metric | PyTorch `best.pt` | ONNX `best.onnx` |
| --- | --- | --- |
| mAP50 | 0.9478 | 0.9464 |
| mAP50-95 | 0.6919 | 0.6938 |
| mAP75 | 0.7700 | 0.7702 |

Precision 0.956 · Recall 0.906 · 1.5 ms/image on an RTX 5060 Laptop GPU.

| class | mAP50 | mAP50-95 | test instances |
| --- | --- | --- | --- |
| X-Intersections | 0.984 | 0.705 | 765 |
| Goalpost | 0.979 | 0.790 | 1081 |
| K1 (robots) | 0.972 | 0.798 | 2213 |
| Ball | 0.940 | 0.765 | 498 |
| L-Intersections | 0.932 | 0.626 | 2708 |
| Penalty Marks | 0.921 | 0.617 | 547 |
| T-Intersections | 0.907 | 0.544 | 2029 |

T/L/X-Intersections are the weakest classes — they are visually near-identical
patches distinguished only by line topology, and the residual error is
confusion among those three.

## ONNX interface

```
input   images   [1, 3, 640, 640]  float32, RGB, 0-1, letterboxed
output  output0  [1, 11, 8400]     cx,cy,w,h (pixels) + 7 class scores
```

No NMS is baked in — apply it downstream. Class order is the `names` list in
`hslvision.yaml` (`0 Ball … 6 X-Intersections`).

## Reproducing

```bash
uv venv --python 3.12 .venv                 # system python 3.14 has no torch wheels
uv pip install -r requirements.txt
hf download whirlwind-ams/hslvision --repo-type dataset --local-dir raw   # ~6 GB
python convert.py                            # parquet -> YOLO layout + hslvision.yaml
python train.py                              # train, eval on test, export ONNX
```

`setup.sh` automates the env + dataset steps with retries.

### Dataset conversion notes

HSLVision ships as HF parquet with embedded images, not a YOLO tree, so
`convert.py` does the translation. Two details that matter:

- **Category ids are 1-indexed**, with `0` reserved for `background`. They are
  remapped to YOLO's 0-based classes (`cat - 1`).
- **Boxes are COCO absolute top-left `xywh`.** This is not stated on the dataset
  card; it was verified empirically by `check_bbox.py` (`w*h` matches the
  dataset's own `area` field on every sampled box, and `xyxy` yields negative
  areas) and confirmed visually with `verify_labels.py`.

Conversion reconciles exactly with the dataset card: 6,068 images / 65,285 boxes
(train 4,247 / valid 910 / test 911), zero dropped boxes, zero filename
collisions. 1,957 train boxes are clipped to the frame (objects running
off-image) and 77 train images have no annotations (usable as background
negatives).

Class distribution is skewed — L-Intersections (17,918) and K1 (14,996) dominate
while Ball (3,344) and Penalty Marks (3,820) are ~5x rarer.

## Training setup

yolo26n · 100 epochs · imgsz 640 · batch 32 · AdamW (auto, lr 9.09e-4) · AMP ·
RAM-cached images · patience 25. Ran 43 min on an RTX 5060 Laptop (8 GB), peak
4.3 GB VRAM.

Early stopping never triggered — all 100 epochs ran and validation was still
improving at the end, so the ceiling is model capacity rather than overfitting.
Validation box/cls/dfl losses fell monotonically alongside train losses; the
val-loss slope never turned positive (`overfit_check.py` monitors this). Test
mAP50-95 (0.692) came in slightly *above* validation (0.684), consistent with
genuine generalization.

Export uses `best.pt` (peak-fitness checkpoint), not `last.pt`, and the test
split was untouched during training and model selection.

> Reproducibility caveat: `cache='ram'` is used for speed, which ultralytics
> warns is non-deterministic even with `seed=0`. Use `cache='disk'` in
> `train.py` for bit-reproducible runs.

## Scripts

| file | purpose |
| --- | --- |
| `setup.sh` | env + dataset download, with retries |
| `convert.py` | HF parquet -> YOLO images/labels, with reconciliation counts |
| `check_bbox.py` | proves the bbox convention against the `area` field |
| `verify_labels.py` | renders labels onto images; audits coord ranges |
| `train.py` | train, evaluate on test split, export ONNX |
| `overfit_check.py` | train/val loss-slope diagnostic, overfitting verdict |
| `verify_onnx.py` | ONNX vs PyTorch equivalence (per-channel tolerances) |

### ONNX verification

`verify_onnx.py` compares raw outputs on identical preprocessed input, using
tolerances appropriate to each channel group — box channels are pixel-scale
(0-640) while class scores are (0-1), so a single absolute tolerance is
meaningless. Result: box `max_rel = 5e-5`, class scores `max_abs = 4e-6`, plus
`onnx.checker` pass. The full-split ONNX eval in the table above is the
end-to-end confirmation.

Note that comparing `.pt` against `.onnx` via `model.predict()` shows spurious
disagreement: ultralytics letterboxes `.pt` rectangularly (`auto=True`) but the
ONNX has a fixed 640x640 input, so they are different inferences. Compare at
fixed resolution.

## Not in git

`raw/` (6 GB parquet), `dataset/` (1.6 GB converted images), `.venv/` (5.6 GB)
and pretrained `yolo26n.pt` are excluded — all are regenerable via the commands
above.

## License

Dataset is CC BY 4.0 (whIRLwind Amsterdam / University of Amsterdam); cite
`catarrinho2026hslvision`. Model weights here inherit that attribution
requirement.
