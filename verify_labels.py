"""Render YOLO labels back onto images as a visual end-to-end check."""
import random, sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
NAMES = ["Ball", "Goalpost", "K1", "L-Int", "PenMark", "T-Int", "X-Int"]
COLORS = ["#ff3b30", "#34c759", "#007aff", "#ffcc00",
          "#ff2d55", "#5ac8fa", "#af52de"]

split = sys.argv[1] if len(sys.argv) > 1 else "train"
img_dir = ROOT / "dataset" / "images" / split
lbl_dir = ROOT / "dataset" / "labels" / split

# prefer images that contain a Ball (class 0) and a Goalpost (class 1)
cands = []
for p in sorted(img_dir.glob("*.jpg"))[:400]:
    t = (lbl_dir / f"{p.stem}.txt").read_text().split("\n")
    cls = {int(l.split()[0]) for l in t if l.strip()}
    if {0, 1} <= cls:
        cands.append(p)
random.seed(0)
picks = (cands or sorted(img_dir.glob("*.jpg"))[:4])[:4]

out = ROOT / "label_check"
out.mkdir(exist_ok=True)
bad = 0
for p in picks:
    im = Image.open(p).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    for line in (lbl_dir / f"{p.stem}.txt").read_text().split("\n"):
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        c = int(c); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            bad += 1
        x0, y0 = (cx - bw / 2) * W, (cy - bh / 2) * H
        x1, y1 = (cx + bw / 2) * W, (cy + bh / 2) * H
        d.rectangle([x0, y0, x1, y1], outline=COLORS[c], width=3)
        d.text((x0 + 3, max(0, y0 - 12)), NAMES[c], fill=COLORS[c])
    im.save(out / f"{p.stem}_boxes.jpg", quality=90)
    print("wrote", out / f"{p.stem}_boxes.jpg", f"({W}x{H})")
print("out-of-range coords:", bad)

# global range audit across the whole split
mn, mx, n = 1e9, -1e9, 0
for lp in lbl_dir.glob("*.txt"):
    for line in lp.read_text().split("\n"):
        if line.strip():
            v = list(map(float, line.split()[1:]))
            mn, mx, n = min(mn, min(v)), max(mx, max(v)), n + 1
print(f"audited {n} boxes in {split}: min={mn:.4f} max={mx:.4f}")
