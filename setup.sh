#!/bin/bash
# Serialized setup: python deps first, then the 6.2GB dataset. Both retried.
cd "$(dirname "$0")"
export UV_HTTP_TIMEOUT=600
export VIRTUAL_ENV=.venv

echo "===== STAGE 1: python deps ====="
ok=0
for i in 1 2 3 4 5; do
  echo "--- pip attempt $i ---"
  if uv pip install torch torchvision ultralytics onnx onnxruntime onnxslim \
       "huggingface_hub[hf_transfer]"; then
    ok=1; break
  fi
  echo "attempt $i failed, retrying"
  sleep 10
done
[ "$ok" = 1 ] || { echo "STAGE1_FAILED"; exit 1; }
.venv/bin/python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" \
  || { echo "STAGE1_IMPORT_FAILED"; exit 1; }
echo "STAGE1_OK"

echo "===== STAGE 2: dataset ====="
export HF_HUB_ENABLE_HF_TRANSFER=1
ok=0
for i in 1 2 3 4 5 6 7 8; do
  echo "--- download attempt $i ---"
  if .venv/bin/hf download whirlwind-ams/hslvision --repo-type dataset --local-dir raw; then
    ok=1; break
  fi
  echo "attempt $i failed ($(du -sh raw | cut -f1) local), retrying"
  sleep 15
done
[ "$ok" = 1 ] || { echo "STAGE2_FAILED"; exit 1; }
n=$(ls raw/data/*.parquet 2>/dev/null | wc -l)
echo "parquet files: $n"
[ "$n" -eq 13 ] || { echo "STAGE2_INCOMPLETE"; exit 1; }
echo "ALL_SETUP_OK"
