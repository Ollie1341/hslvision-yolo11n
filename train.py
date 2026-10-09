"""Train yolo26m on HSLVision, eval on the held-out test split, export ONNX."""
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = str(ROOT / "hslvision.yaml")
RUN = "yolo26m-hslvision"

if __name__ == "__main__":
    # Changed the model to yolo26m
    model = YOLO("yolo26m.pt")

    model.train(
        data=DATA,
        epochs=500,
        imgsz=640,
        batch=32,
        device=0,
        # Increased the workers to 16 for faster data loading
        workers=16,
        cache="ram",          # ~6GB, plenty of headroom; avoids IO bottleneck
        amp=True,
        seed=0,
        patience=25,
        project=str(ROOT / "runs"),
        name=RUN,
        exist_ok=True,
        plots=True,
        val=True,
    )

    best = ROOT / "runs" / RUN / "weights" / "best.pt"
    print(f"\n===== BEST WEIGHTS: {best} =====", flush=True)

    m = YOLO(str(best))

    print("\n===== TEST SPLIT EVAL =====", flush=True)
    tm = m.val(data=DATA, split="test", imgsz=640, batch=32, device=0,
               project=str(ROOT / "runs"), name="test-eval", exist_ok=True,
               plots=True)
    print(f"TEST mAP50-95 = {tm.box.map:.4f}", flush=True)
    print(f"TEST mAP50    = {tm.box.map50:.4f}", flush=True)
    print(f"TEST mAP75    = {tm.box.map75:.4f}", flush=True)
    print("TEST per-class mAP50-95:", flush=True)
    for i, c in enumerate(tm.box.ap_class_index):
        print(f"   {tm.names[c]:18s} {tm.box.maps[c]:.4f}", flush=True)

    print("\n===== ONNX EXPORT =====", flush=True)
    onnx = m.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
    print(f"ONNX_PATH={onnx}", flush=True)
    print("TRAIN_PIPELINE_DONE", flush=True)
