#!/usr/bin/env bash
# ===========================================================================
# 一键运行脚本：自动创建 venv、安装依赖、启动训练
# 用法:
#   ./run.sh            默认训练 (1小时)
#   ./run.sh 30m        训练 30 分钟
#   ./run.sh 2h         训练 2 小时
#   ./run.sh setup      仅安装依赖，不训练
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# ---------- 创建 venv ----------
setup_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "[venv] 创建虚拟环境..."
        python3 -m venv "$VENV_DIR"
    fi

    echo "[venv] 升级 pip..."
    "$PIP" install --upgrade pip -q

    echo "[venv] 安装依赖..."
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS: 跳过 cupy (仅 NVIDIA GPU)
        "$PIP" install -q -r <(grep -v '^cupy' requirements.txt)
    else
        "$PIP" install -q -r requirements.txt
    fi

    echo "[venv] 依赖就绪 ✓"
}

# ---------- 主流程 ----------
main() {
    setup_venv

    if [[ "${1:-}" == "setup" ]]; then
        echo "[完成] 仅安装依赖，跳过训练"
        exit 0
    fi

    echo "[启动] 开始训练..."
    echo ""
    exec bash train.sh "${1:-}"
}

main "$@"
