#!/usr/bin/env bash
# Simple wrapper to run KrakenHMM trader as a service from project root.
# Usage: ./scripts/run_trader_service.sh start|stop|status
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venv"
PY="$VENV/bin/python"
PY_UNBUFFERED="$VENV/bin/python -u"
LOG_DIR="$PROJECT_DIR/logs"
STATE_FILE="$PROJECT_DIR/state/trader_state.json"
PID_FILE="$PROJECT_DIR/state/trader.pid"
READY_FILE="$PROJECT_DIR/state/api.ready"
PORT=8765
ARGS=("--simulate" "--enable-api" "--http-port" "$PORT" "--trade-log-dir" "$LOG_DIR" "--state-file" "$STATE_FILE")
case "$1" in
start)
  mkdir -p "$LOG_DIR" "$PROJECT_DIR/state"
  if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") >/dev/null 2>&1; then
    echo "Already running (pid $(cat $PID_FILE))."
    exit 0
  fi
  nohup $PY_UNBUFFERED "$PROJECT_DIR/run_trader.py" "${ARGS[@]}" > "$LOG_DIR/trader.out" 2>&1 &
  echo $! > "$PID_FILE"
  echo "Started trader (pid $(cat $PID_FILE))."
  ;;
stop)
  if [ -f "$PID_FILE" ]; then
    kill $(cat "$PID_FILE") || true
    rm -f "$PID_FILE"
    [ -f "$READY_FILE" ] && rm -f "$READY_FILE"
    echo "Stopped trader."
  else
    echo "No pid file, trader may not be running."
  fi
  ;;
status)
  if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") >/dev/null 2>&1; then
    echo "Trader running (pid $(cat $PID_FILE))."
    [ -f "$READY_FILE" ] && echo "API ready file present: $READY_FILE"
  else
    echo "Trader not running."
  fi
  ;;
*)
  echo "Usage: $0 {start|stop|status}"
  exit 1
  ;;
esac
