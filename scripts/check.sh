#!/usr/bin/env bash
# TalentGraph deterministic local verification gate (Harness feedback loop)
#
# Usage:
#   ./scripts/check.sh          # Full gate: ruff + mypy + pytest(--cov) + benchmark
#   ./scripts/check.sh fast     # Fast TDD loop: ruff + pytest
#   ./scripts/check.sh lint     # Ruff lint only
#   ./scripts/check.sh type     # Mypy type check only
#   ./scripts/check.sh test     # Pytest + coverage only
#   ./scripts/check.sh bench    # Core CPU embedding & 16-dim scoring benchmark only
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$REPO_ROOT/backend"
VENV_BIN="$BACKEND_DIR/.venv/bin"

MODE="${1:-all}"

if [ ! -x "$VENV_BIN/python" ]; then
  echo "[ERROR] Virtual environment not found at $BACKEND_DIR/.venv" >&2
  echo "Run:" >&2
  echo "  uv venv backend/.venv --python 3.10" >&2
  echo "  uv pip install --python backend/.venv/bin/python -e \"backend[dev]\"" >&2
  exit 2
fi

PYTHON="$VENV_BIN/python"
RUFF="$VENV_BIN/ruff"
MYPY="$VENV_BIN/mypy"
PYTEST="$VENV_BIN/pytest"

FAILURES=()

run_step() {
  local name="$1"
  shift
  echo ""
  echo "============================================================"
  echo "==> [$name]"
  echo "============================================================"
  if "$@"; then
    echo "[PASS] $name"
  else
    local rc=$?
    echo "[FAIL] $name (exit $rc)"
    FAILURES+=("$name")
  fi
}

run_lint() {
  (cd "$BACKEND_DIR" && "$RUFF" check src configs tests ../cli.py)
}

run_type() {
  (cd "$BACKEND_DIR" && "$MYPY" src configs)
}

run_test_fast() {
  (cd "$BACKEND_DIR" && "$PYTEST" tests/ -q)
}

run_test_cov() {
  (cd "$BACKEND_DIR" && "$PYTEST" tests/ --cov=src --cov-report=term-missing)
}

run_bench() {
  (cd "$REPO_ROOT" && "$PYTHON" cli.py benchmark)
}

case "$MODE" in
  all)
    run_step "ruff lint" run_lint
    run_step "mypy typecheck" run_type
    run_step "pytest + coverage" run_test_cov
    run_step "performance benchmark" run_bench
    ;;
  fast)
    run_step "ruff lint" run_lint
    run_step "pytest (fast)" run_test_fast
    ;;
  lint)
    run_step "ruff lint" run_lint
    ;;
  type)
    run_step "mypy typecheck" run_type
    ;;
  test)
    run_step "pytest + coverage" run_test_cov
    ;;
  bench)
    run_step "performance benchmark" run_bench
    ;;
  *)
    echo "Unknown mode: $MODE (expected: all | fast | lint | type | test | bench)" >&2
    exit 3
    ;;
esac

echo ""
echo "============================================================"
if [ "${#FAILURES[@]}" -eq 0 ]; then
  echo "[ALL PASS] TalentGraph verification gate ($MODE) succeeded."
  echo "============================================================"
  exit 0
else
  echo "[GATE FAILED] Failed checks (${#FAILURES[@]}): ${FAILURES[*]}"
  echo "============================================================"
  exit 1
fi
