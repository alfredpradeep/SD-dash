#!/usr/bin/env bash
# SHIELD v2 launcher — production-grade entrypoint.
#
# Usage:
#   ./run_shield_v2.sh                    # boot server on :8765 (default)
#   ./run_shield_v2.sh serve 9000         # boot server on :9000
#   ./run_shield_v2.sh --port 9000        # boot server on :9000
#   ./run_shield_v2.sh test               # run full pytest suite
#   ./run_shield_v2.sh install            # install dependencies only
#   SHIELD_PORT=9000 ./run_shield_v2.sh   # env-var override

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export SHIELD_HOST="${SHIELD_HOST:-0.0.0.0}"
# Default to 8765 — 8000 is frequently occupied on dev machines.
export SHIELD_PORT="${SHIELD_PORT:-8765}"

PY="${PYTHON:-python3}"

install_deps() {
    echo "→ Installing SHIELD v2 dependencies..."
    $PY -m pip install --quiet --break-system-packages \
        "fastapi>=0.110" "uvicorn[standard]>=0.27" \
        "pydantic>=2.5" "numpy>=1.26" "scipy>=1.11" \
        "scikit-learn>=1.3" "jinja2>=3.1" "httpx>=0.25" \
        "pytest>=7.4" "pytest-asyncio>=0.23" || true
}

port_is_free() {
    local port="$1"
    # portable: try lsof then python fallback
    if command -v lsof >/dev/null 2>&1; then
        ! lsof -iTCP:"$port" -sTCP:LISTEN -P -n >/dev/null 2>&1
    else
        $PY - <<EOF
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("0.0.0.0", $port))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
EOF
    fi
}

find_free_port() {
    local start="$1"
    for offset in 0 1 2 3 4 5 6 7 8 9 10; do
        local candidate=$((start + offset))
        if port_is_free "$candidate"; then
            echo "$candidate"; return 0
        fi
    done
    # Fall back to an ephemeral port picked by the kernel
    $PY - <<'EOF'
import socket
s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()
EOF
}

run_tests() {
    install_deps
    echo "→ Running SHIELD v2 test suite..."
    $PY -m pytest shield_v2/tests/ -v --tb=short "$@"
}

run_server() {
    install_deps
    if ! port_is_free "$SHIELD_PORT"; then
        local new_port
        new_port="$(find_free_port "$SHIELD_PORT")"
        echo "⚠  Port $SHIELD_PORT is already in use — falling back to $new_port"
        export SHIELD_PORT="$new_port"
    fi
    echo "→ SHIELD v2 booting on http://${SHIELD_HOST}:${SHIELD_PORT}"
    echo "   Open: http://127.0.0.1:${SHIELD_PORT}/"
    exec $PY -m shield_v2.run
}

# Parse command + optional positional port / --port flag
CMD="${1:-serve}"
case "$CMD" in
    test|tests)
        shift
        run_tests "$@"
        ;;
    install)
        install_deps
        ;;
    --port)
        shift
        export SHIELD_PORT="${1:-$SHIELD_PORT}"
        run_server
        ;;
    serve|"")
        [[ "$CMD" == "serve" ]] && shift || true
        # Optional positional port argument: ./run_shield_v2.sh 9000
        if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
            export SHIELD_PORT="$1"; shift
        elif [[ "${1:-}" == "--port" ]]; then
            shift; export SHIELD_PORT="${1:-$SHIELD_PORT}"; shift || true
        fi
        run_server
        ;;
    *)
        # Treat a bare numeric first arg as a port
        if [[ "$CMD" =~ ^[0-9]+$ ]]; then
            export SHIELD_PORT="$CMD"
            run_server
        else
            echo "Unknown command: $CMD"
            echo "Usage: $0 [serve [PORT] | test | install | --port PORT]"
            exit 1
        fi
        ;;
esac
