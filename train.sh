#!/usr/bin/env bash
# ===========================================================================
# 训练时长控制脚本
# 用法:
#   ./train.sh            默认训练 (1小时)
#   ./train.sh 30m        训练 30 分钟
#   ./train.sh 1h         训练 1 小时
#   ./train.sh 2h         训练 2 小时
#   ./train.sh 4h         训练 4 小时
#   ./train.sh 6h         训练 6 小时
#   ./train.sh 90m        训练 90 分钟（自定义）
#   ./train.sh 1h30m      自定义 1.5 小时
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 时长 -> epochs 预计算
get_preset_epochs() {
    case "$1" in
        30m|30min) echo 20 ;;
        1h)        echo 39 ;;
        2h)        echo 79 ;;
        4h)        echo 158 ;;
        6h)        echo 237 ;;
        *)         echo "" ;;
    esac
}

# 解析人类可读时长为秒
parse_duration() {
    local input="$1"
    local total_sec=0
    while [[ "$input" =~ ([0-9]+)([mh]) ]]; do
        local num="${BASH_REMATCH[1]}"
        local unit="${BASH_REMATCH[2]}"
        if [[ "$unit" == "h" ]]; then
            total_sec=$((total_sec + num * 3600))
        else
            total_sec=$((total_sec + num * 60))
        fi
        input="${input#*"${BASH_REMATCH[0]}"}"
    done
    echo "$total_sec"
}

# 秒 -> 小时/分钟
human_duration() {
    local sec="$1"
    if ((sec >= 3600)); then
        printf '%dh%02dm' $((sec / 3600)) $((sec % 3600 / 60))
    else
        printf '%dm' $((sec / 60))
    fi
}

main() {
    local label epochs target_sec

    if [[ $# -eq 0 ]]; then
        label="1h"
        epochs=$(get_preset_epochs "$label")
        target_sec=3600
        echo "未指定时长，默认训练 1 小时"
    else
        label="$1"
        epochs=$(get_preset_epochs "$label")
        if [[ -n "$epochs" ]]; then
            target_sec=$((epochs * 91))
        else
            target_sec=$(parse_duration "$label")
            if [[ "$target_sec" -eq 0 ]]; then
                echo "错误: 无法识别的时长 '$label'"
                echo "支持的快捷方式: 30m 1h 2h 4h 6h"
                echo "或自定义: 90m / 1h30m / 2h 等"
                exit 1
            fi
            epochs=$(python3 -c "print(round($target_sec / 91.2))")
        fi
    fi

    local actual_human
    actual_human=$(python3 _duration.py "$epochs")

    echo "============================================"
    echo "  训练时长配置"
    echo "  目标时长:     $(human_duration "$target_sec")"
    echo "  Epochs:       $epochs"
    echo "  预计实际耗时: $actual_human"
    echo "============================================"
    echo ""

    rm -f gpt_chinese.npz

    GPT_EPOCHS="$epochs" python3 train_gpt.py 2>&1
}

main "$@"
