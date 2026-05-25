#!/usr/bin/env bash
set -euo pipefail

# 用法：
#   bash install_vllm.sh
#
# 可选：
#   VLLM_VERSION=0.20.0 bash install_vllm.sh
#   PIP_ARGS="--index-url https://pypi.org/simple" bash install_vllm.sh
#
# 说明：
#   - 不创建虚拟环境
#   - 不安装 torch
#   - 根据当前已安装 torch 选择 vLLM
#   - 用 constraints 锁住当前 torch，防止 pip 改动 torch

PYTHON_BIN="${PYTHON_BIN:-python}"

TORCH_FULL_VERSION="$(
  "$PYTHON_BIN" - <<'PY'
import re
import torch

v = torch.__version__
m = re.match(r"^(\d+\.\d+\.\d+)", v)
if not m:
    raise RuntimeError(f"Cannot parse torch version: {v}")

print(v)
PY
)"

TORCH_BASE_VERSION="$(
  "$PYTHON_BIN" - <<'PY'
import re
import torch

m = re.match(r"^(\d+\.\d+\.\d+)", torch.__version__)
print(m.group(1))
PY
)"

case "$TORCH_BASE_VERSION" in
  2.8.0)
    DEFAULT_VLLM_VERSION="0.11.0"
    ;;
  2.9.0)
    DEFAULT_VLLM_VERSION="0.13.0"
    ;;
  2.9.1)
    DEFAULT_VLLM_VERSION="0.16.0"
    ;;
  2.10.0)
    DEFAULT_VLLM_VERSION="0.19.1"
    ;;
  2.11.0)
    DEFAULT_VLLM_VERSION="0.21.0"
    ;;
  *)
    echo "ERROR: 当前 torch 版本暂未配置对应 vLLM：$TORCH_FULL_VERSION"
    echo "已配置：2.8.0, 2.9.0, 2.9.1, 2.10.0, 2.11.0"
    exit 1
    ;;
esac

VLLM_VERSION="${VLLM_VERSION:-$DEFAULT_VLLM_VERSION}"

CONSTRAINT_FILE="$(mktemp)"
trap 'rm -f "$CONSTRAINT_FILE"' EXIT

cat > "$CONSTRAINT_FILE" <<EOF
torch==$TORCH_FULL_VERSION
EOF

echo "Python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "torch detected: $TORCH_FULL_VERSION"
echo "vLLM selected:  $VLLM_VERSION"
echo "constraint:     torch==$TORCH_FULL_VERSION"
echo

"$PYTHON_BIN" -m pip install --user \
  "vllm==$VLLM_VERSION" \
  --constraint "$CONSTRAINT_FILE" \
  ${PIP_ARGS:-}

echo
"$PYTHON_BIN" - <<'PY'
import torch
import vllm

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
print("vllm:", vllm.__version__)
PY