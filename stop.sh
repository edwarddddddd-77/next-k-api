#!/usr/bin/env bash
# stop.sh — 停止 Next K API 及调度器进程
# 用法：./stop.sh
set -euo pipefail

# ── 路径常量 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="$SCRIPT_DIR/.pid"
API_PID_FILE="$PID_DIR/api.pid"
SCHED_PID_FILE="$PID_DIR/scheduler.pid"

# 优雅关闭超时（秒）
GRACEFUL_TIMEOUT=15

# ── 颜色 ──────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[next-k-api]${NC} $*"; }
warn()  { echo -e "${YELLOW}[next-k-api]${NC} $*"; }
error() { echo -e "${RED}[next-k-api]${NC} $*" >&2; }

# ── 核心：停止单个进程 ────────────────────────────────────────────────────────
# 用法：stop_process <label> <pid_file>
stop_process() {
    local label="$1"
    local pid_file="$2"

    if [[ ! -f "$pid_file" ]]; then
        warn "$label：未找到 PID 文件（$pid_file），跳过。"
        return 0
    fi

    local pid
    pid=$(cat "$pid_file")

    if ! kill -0 "$pid" 2>/dev/null; then
        warn "$label（PID=$pid）已不在运行，清理 PID 文件。"
        rm -f "$pid_file"
        return 0
    fi

    info "$label（PID=$pid）：发送 SIGTERM..."
    kill -TERM "$pid" 2>/dev/null || true

    # 等待优雅退出
    local elapsed=0
    while kill -0 "$pid" 2>/dev/null; do
        if [[ $elapsed -ge $GRACEFUL_TIMEOUT ]]; then
            warn "$label（PID=$pid）在 ${GRACEFUL_TIMEOUT}s 内未退出，强制 SIGKILL..."
            kill -KILL "$pid" 2>/dev/null || true
            break
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    # 最终确认
    if kill -0 "$pid" 2>/dev/null; then
        error "$label（PID=$pid）SIGKILL 后仍在运行，请手动处理：kill -9 $pid"
    else
        info "$label（PID=$pid）已停止。"
    fi

    rm -f "$pid_file"
}

# ── 执行停止 ──────────────────────────────────────────────────────────────────
STOPPED_ANY=false

if [[ -f "$SCHED_PID_FILE" ]]; then
    stop_process "调度器" "$SCHED_PID_FILE"
    STOPPED_ANY=true
fi

if [[ -f "$API_PID_FILE" ]]; then
    stop_process "API" "$API_PID_FILE"
    STOPPED_ANY=true
fi

if ! $STOPPED_ANY; then
    warn "未找到任何 PID 文件（$PID_DIR/*.pid），服务可能未在运行。"
    exit 0
fi

# ── 清理空 PID 目录 ───────────────────────────────────────────────────────────
if [[ -d "$PID_DIR" ]] && [[ -z "$(ls -A "$PID_DIR" 2>/dev/null)" ]]; then
    rmdir "$PID_DIR" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Next K API 已停止${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "  启动服务：./start.sh"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
