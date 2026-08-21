"""Empirically determine the bbox convention by testing each hypothesis
against the dataset's own `area` field and the image bounds."""
import glob, io, sys
import pyarrow.parquet as pq
from PIL import Image

f = sorted(glob.glob("raw/data/*.parquet"))
if not f:
    sys.exit("no parquet yet")
print("using", f[0])

pf = pq.ParquetFile(f[0])
rows = next(pf.iter_batches(batch_size=12)).to_pylist()

score = {"xywh": 0, "xyxy": 0, "cxcywh": 0}
oob = {"xywh": 0, "xyxy": 0, "cxcywh": 0}
n = 0
norm_suspect = 0

for r in rows:
    im = Image.open(io.BytesIO(r["image"]["bytes"]))
    W, H = im.size
    print(f"\n{r['filename']}  meta={r['width']}x{r['height']}  decoded={W}x{H}"
          f"  nobj={len(r['objects']['bbox'])}")
    for bbox, cat, area in list(zip(r["objects"]["bbox"], r["objects"]["category"],
                                    r["objects"]["area"]))[:4]:
        a, b, c, d = bbox
        if max(a, b, c, d) <= 1.0:
            norm_suspect += 1
        cands = {
            "xywh":   c * d,                          # w*h
            "xyxy":   (c - a) * (d - b),               # (x1-x0)*(y1-y0)
            "cxcywh": c * d,                           # same area as xywh
        }
        # geometric validity: does the box stay inside the image?
        fits = {
            "xywh":   a >= 0 and b >= 0 and a + c <= W + 1 and b + d <= H + 1,
            "xyxy":   c > a and d > b and c <= W + 1 and d <= H + 1,
            "cxcywh": a - c / 2 >= -1 and b - d / 2 >= -1
                      and a + c / 2 <= W + 1 and b + d / 2 <= H + 1,
        }
        best = None
        for k, v in cands.items():
            rel = abs(v - area) / max(area, 1e-6)
            if rel < 0.02:
                score[k] += 1
                if best is None:
                    best = k
            if not fits[k]:
                oob[k] += 1
        n += 1
        print(f"   bbox={[round(x,1) for x in bbox]} area={area:.1f} cat={cat} "
              f"| w*h={cands['xywh']:.1f} (x1-x0)(y1-y0)={cands['xyxy']:.1f} "
              f"| fits={ {k:int(v) for k,v in fits.items()} }")

print(f"\n--- {n} boxes tested ---")
print("area matches:", score)
print("out-of-bounds counts (lower=better):", oob)
print("boxes with all coords <=1.0 (would imply normalized):", norm_suspect)
