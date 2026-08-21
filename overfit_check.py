"""Diagnose over/under-fitting from ultralytics results.csv."""
import csv, sys
from pathlib import Path

p = Path(__file__).parent / "runs" / "yolo11n-hslvision" / "results.csv"
rows = list(csv.DictReader(p.open()))
if not rows:
    sys.exit("no rows yet")

def f(r, k): return float(r[k])

ep = [int(f(r, "epoch")) for r in rows]
tb = [f(r, "train/box_loss") for r in rows]
tc = [f(r, "train/cls_loss") for r in rows]
td = [f(r, "train/dfl_loss") for r in rows]
vb = [f(r, "val/box_loss") for r in rows]
vc = [f(r, "val/cls_loss") for r in rows]
vd = [f(r, "val/dfl_loss") for r in rows]
m50 = [f(r, "metrics/mAP50(B)") for r in rows]
m95 = [f(r, "metrics/mAP50-95(B)") for r in rows]

# ultralytics "fitness" = 0.1*mAP50 + 0.9*mAP50-95 (drives best.pt + early stop)
fit = [0.1 * a + 0.9 * b for a, b in zip(m50, m95)]
bi = max(range(len(fit)), key=lambda i: fit[i])
n = len(rows)

print(f"epochs completed: {n}  (of 100)")
print(f"best fitness epoch: {ep[bi]}  fitness={fit[bi]:.4f} "
      f"mAP50={m50[bi]:.4f} mAP50-95={m95[bi]:.4f}")
print(f"epochs since best: {ep[-1] - ep[bi]}  (early-stop patience=25)")
print()
print(f"{'ep':>4} {'trainB':>7} {'valB':>7} {'trainC':>7} {'valC':>7} "
      f"{'trainD':>7} {'valD':>7} {'mAP50':>7} {'mAP95':>7}")
step = max(1, n // 14)
idx = sorted(set(list(range(0, n, step)) + [n - 1, bi]))
for i in idx:
    star = " *best" if i == bi else ""
    print(f"{ep[i]:>4} {tb[i]:>7.4f} {vb[i]:>7.4f} {tc[i]:>7.4f} {vc[i]:>7.4f} "
          f"{td[i]:>7.4f} {vd[i]:>7.4f} {m50[i]:>7.4f} {m95[i]:>7.4f}{star}")

# trend over the last window: are val losses rising while train falls?
w = min(10, max(3, n // 4))
def slope(a):
    ys = a[-w:]; xs = list(range(len(ys))); k = len(ys)
    mx = sum(xs) / k; my = sum(ys) / k
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den

print(f"\n--- trend over last {w} epochs (slope per epoch) ---")
verdict = []
for name, tr, va in (("box", tb, vb), ("cls", tc, vc), ("dfl", td, vd)):
    st, sv = slope(tr), slope(va)
    tag = ""
    if sv > 1e-4 and st < 0:
        tag = "  <-- val rising while train falls (overfit signal)"
        verdict.append(name)
    print(f"  {name}: train {st:+.5f}/ep   val {sv:+.5f}/ep{tag}")
sf = slope(fit)
print(f"  fitness: {sf:+.5f}/ep" + ("  <-- declining" if sf < -1e-4 else ""))

print("\n--- generalization gap (val - train) ---")
for name, tr, va in (("box", tb, vb), ("cls", tc, vc), ("dfl", td, vd)):
    print(f"  {name}: first={va[0]-tr[0]:+.4f}  now={va[-1]-tr[-1]:+.4f}")

print("\nVERDICT:", end=" ")
if verdict and sf < 0:
    print(f"OVERFITTING on {', '.join(verdict)} and fitness declining.")
elif verdict:
    print(f"early overfit signal on {', '.join(verdict)}, but fitness still improving.")
elif sf > 1e-4:
    print("still improving, no overfitting.")
else:
    print("plateauing, no clear overfitting.")
