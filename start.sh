#!/usr/bin/env bash
# start.sh — 启动 Next K API（API 进程 + 可选独立调度器进程）
# 用法：./start.sh
# 环境变量覆盖：PORT=9000 ./start.sh  /  NEXT_K_EMBED_SCHEDULER=1 ./start.sh
set -euo pipefail

# ── 路径常量 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
PID_DIR="$SCRIPT_DIR/.pid"
LOG_DIR="$SCRIPT_DIR/logs"
API_PID_FILE="$PID_DIR/api.pid"
SCHED_PID_FILE="$PID_DIR/scheduler.pid"
API_LOG="$LOG_DIR/api.log"
SCHED_LOG="$LOG_DIR/scheduler.log"
ENV_FILE="$SCRIPT_DIR/.env.oi"
ENV_EXAMPLE="$SCRIPT_DIR/.env.oi.example"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# ── 颜色 ──────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[next-k-api]${NC} $*"; }
warn()  { echo -e "${YELLOW}[next-k-api]${NC} $*"; }
error() { echo -e "${RED}[next-k-api]${NC} $*" >&2; }

# ── 辅助：检查进程是否存活 ───────────────────────────────────────────────────
is_running() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] || return 1
    local pid
    pid=$(cat "$pid_file")
    kill -0 "$pid" 2>/dev/null
}

# ── 0. 防止重复启动 ───────────────────────────────────────────────────────────
if is_running "$API_PID_FILE"; then
    warn "API 进程已在运行（PID=$(cat "$API_PID_FILE")），跳过启动。"
    warn "如需重启，请先运行：./stop.sh"
    exit 0
fi

# ── 1. 检查 Python 版本 ───────────────────────────────────────────────────────
info "检查 Python 版本..."

PYTHON_BIN=""
for py in python3.11 python3 python; do
    if command -v "$py" &>/dev/null; then
        ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [[ "$major" -eq 3 && "$minor" -ge 11 ]]; then
            PYTHON_BIN="$py"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    error "未找到 Python 3.11+。请先安装 Python 3.11 或更高版本。"
    exit 1
fi
info "使用 Python: $PYTHON_BIN ($(${PYTHON_BIN} --version 2>&1))"

# ── 2. 创建 / 复用虚拟环境 ───────────────────────────────────────────────────
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    info "创建虚拟环境：$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    info "复用已有虚拟环境：$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
PYTHON_VENV="$VENV_DIR/bin/python"

# ── 3. 安装 / 更新依赖 ────────────────────────────────────────────────────────
info "安装依赖（$REQUIREMENTS）..."
"$PYTHON_VENV" -m pip install --quiet --upgrade pip
"$PYTHON_VENV" -m pip install --quiet -r "$REQUIREMENTS"
info "依赖安装完成。"

# ── 4. 准备 .env.oi ───────────────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f "$ENV_EXAMPLE" ]]; then
        warn ".env.oi 不存在，从 .env.oi.example 复制..."
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        warn "请编辑 $ENV_FILE 并设置 NEXT_K_MAINTENANCE_TOKEN 等必要变量后重新启动。"
    else
        warn ".env.oi 和 .env.oi.example 均不存在，将使用默认配置启动。"
    fi
fi

# 读取 .env.oi（setdefault 语义：不覆盖命令行已设置的变量）
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        key=$(echo "$key" | xargs)
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ "$key" != *[[:space:]]* ]] || continue
        # Strip inline comment (anything after the first whitespace+# sequence)
        value=$(echo "$value" | sed 's/[[:space:]]*#.*//' | xargs)
        if [[ -z "${!key+x}" ]]; then
            export "$key"="$value"
        fi
    done < <(grep -v '^\s*#' "$ENV_FILE" | grep '=')
fi

# ── 5. 读取关键参数 ───────────────────────────────────────────────────────────
PORT="${PORT:-8000}"
EMBED_SCHEDULER="${NEXT_K_EMBED_SCHEDULER:-}"

# 判断是否为单进程模式
is_embed=false
if [[ "$EMBED_SCHEDULER" == "1" || "$EMBED_SCHEDULER" == "true" || \
      "$EMBED_SCHEDULER" == "yes" || "$EMBED_SCHEDULER" == "on" ]]; then
    is_embed=true
fi

# ── 6. 创建目录 ───────────────────────────────────────────────────────────────
mkdir -p "$PID_DIR" "$LOG_DIR"

# ── 7. 启动 API 进程 ──────────────────────────────────────────────────────────
info "启动 API（端口 $PORT）..."
nohup "$PYTHON_VENV" -m uvicorn main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    >> "$API_LOG" 2>&1 &
API_PID=$!
echo "$API_PID" > "$API_PID_FILE"
info "API 进程已启动（PID=$API_PID），日志：$API_LOG"

# ── 8. 等待 API 就绪 ──────────────────────────────────────────────────────────
info "等待 API 就绪..."
WAIT_MAX=30
WAIT_COUNT=0
while [[ $WAIT_COUNT -lt $WAIT_MAX ]]; do
    if curl -sf "http://localhost:${PORT}/api/health" >/dev/null 2>&1; then
        info "API 就绪：http://localhost:${PORT}"
        break
    fi
    if ! kill -0 "$API_PID" 2>/dev/null; then
        error "API 进程意外退出。请检查日志：$API_LOG"
        rm -f "$API_PID_FILE"
        exit 1
    fi
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

if [[ $WAIT_COUNT -ge $WAIT_MAX ]]; then
    warn "API 未在 ${WAIT_MAX}s 内响应，可能仍在加载中。请检查：$API_LOG"
fi

# ── 9. 单进程模式下跳过独立调度器 ────────────────────────────────────────────
if $is_embed; then
    info "单进程模式（NEXT_K_EMBED_SCHEDULER=1）：调度器已内嵌 API 进程，无需额外启动。"
else
    # ── 10. 双进程模式：启动独立调度器 ───────────────────────────────────────
    info "双进程模式：启动独立调度器..."
    nohup "$PYTHON_VENV" "$SCRIPT_DIR/scheduler_main.py" \
        >> "$SCHED_LOG" 2>&1 &
    SCHED_PID=$!
    echo "$SCHED_PID" > "$SCHED_PID_FILE"
    info "调度器进程已启动（PID=$SCHED_PID），日志：$SCHED_LOG"
fi

# ── 11. 启动摘要 ──────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "${GREEN}  Next K API 启动成功${NC}"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo -e "  API 地址     : http://localhost:${PORT}"
echo -e "  Swagger 文档 : http://localhost:${PORT}/docs"
echo -e "  健康检查     : http://localhost:${PORT}/api/health"
echo -e "  ZCT 看板     : http://localhost:${PORT}/dashboard/zct-vwap"
echo -e "  API 日志     : $API_LOG"
if ! $is_embed; then
    echo -e "  调度器日志   : $SCHED_LOG"
fi
echo -e "  停止服务     : ./stop.sh"
echo -e "${GREEN}══════════════════════════════════════════${NC}"
echo ""
