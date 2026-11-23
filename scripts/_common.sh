set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

activate_venv() {
    if [[ -f .venv/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
        set +u
        source .venv/bin/activate
        set -u
    fi
}
activate_venv
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

ft() { python -m fragtrust.cli "$@"; }
step() { printf '\n==> [%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
trap 'rc=$?; echo "FAILED (exit $rc): $BASH_COMMAND" >&2' ERR
