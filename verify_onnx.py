"""Rigorous ONNX-vs-PyTorch equivalence check with channel-appropriate tolerances."""
import numpy as np, onnxruntime as ort, torch, cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

ROOT = Path(__file__).parent
W = ROOT / "runs" / "yolo11n-hslvision" / "weights"
sess = ort.InferenceSession(str(W / "best.onnx"), providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
yolo = YOLO(str(W / "best.pt"))
yolo.model.float().eval()

imgs = sorted((ROOT / "dataset" / "images" / "test").glob("*.jpg"))[:12]
box_abs = box_rel = cls_abs = 0.0

for p in imgs:
    im = cv2.imread(str(p))
    x = LetterBox((640, 640), auto=False)(image=im)[:, :, ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x[None].astype(np.float32) / 255.0)
    o_on = sess.run(None, {name: x})[0]
    with torch.no_grad():
        r = yolo.model(torch.from_numpy(x))
    o_pt = (r[0] if isinstance(r, (list, tuple)) else r).numpy()

    b_on, b_pt = o_on[:, :4], o_pt[:, :4]        # xywh in pixels (0..640)
    c_on, c_pt = o_on[:, 4:], o_pt[:, 4:]        # class scores (0..1)
    box_abs = max(box_abs, np.abs(b_on - b_pt).max())
    box_rel = max(box_rel, (np.abs(b_on - b_pt) / np.maximum(np.abs(b_pt), 1e-6)).max())
    cls_abs = max(cls_abs, np.abs(c_on - c_pt).max())

print("=== raw output equivalence over 12 test images ===")
print(f"box channels (px, range 0-640): max_abs={box_abs:.3e}  max_rel={box_rel:.3e}")
print(f"cls channels (prob, range 0-1): max_abs={cls_abs:.3e}")
print("box PASS (rel<1e-3):", box_rel < 1e-3)
print("cls PASS (abs<1e-3):", cls_abs < 1e-3)

# --- detection-level agreement, matching by IoU ---
def iou(a, b):
    x1 = np.maximum(a[0], b[:, 0]); y1 = np.maximum(a[1], b[:, 1])
    x2 = np.minimum(a[2], b[:, 2]); y2 = np.minimum(a[3], b[:, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[..., 2] - z[..., 0]) * (z[..., 3] - z[..., 1])
    return i / (ar(a) + ar(b) - i + 1e-9)

yo = YOLO(str(W / "best.onnx"), task="detect")
print("\n=== detection agreement (conf=0.25, IoU>0.9 match) ===")
tot_m = tot_u = 0; worst = 0.0; near = 0
for p in imgs[:8]:
    a = yolo.predict(p, imgsz=640, conf=0.25, device="cpu", verbose=False)[0].boxes
    b = yo.predict(p, imgsz=640, conf=0.25, verbose=False)[0].boxes
    A = a.xyxy.cpu().numpy(); B = b.xyxy.cpu().numpy()
    ca, cb = a.conf.cpu().numpy(), b.conf.cpu().numpy()
    used = set(); matched = 0; dconf = 0.0
    for i in range(len(A)):
        if len(B) == 0: break
        v = iou(A[i], B); j = int(v.argmax())
        if v[j] > 0.9 and j not in used and int(a.cls[i].item()) == int(b.cls[j].item()):
            used.add(j); matched += 1; dconf = max(dconf, abs(ca[i] - cb[j]))
    unmatched = [float(c) for k, c in enumerate(ca) if k not in range(0)] # placeholder
    um = (len(A) - matched) + (len(B) - matched)
    # how close to the 0.25 cutoff are the unmatched ones?
    lone = [float(c) for c in np.concatenate([ca, cb]) if abs(c - 0.25) < 0.05]
    near += len([c for c in lone])
    tot_m += matched; tot_u += um; worst = max(worst, dconf)
    print(f"  {p.name[:44]:44s} pt={len(A):3d} onnx={len(B):3d} matched={matched:3d} "
          f"unmatched={um} max_dconf={dconf:.2e}")

print(f"\ntotal matched={tot_m} unmatched={tot_u} "
      f"({100*tot_m/(tot_m+tot_u/2+1e-9):.1f}% agreement)")
print(f"max confidence delta on matched detections: {worst:.3e}")
print(f"detections within 0.05 of the 0.25 cutoff: {near}  <- these can flip")
