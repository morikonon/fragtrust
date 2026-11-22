set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -f .venv/bin/activate && -z "${VIRTUAL_ENV:-}" ]]; then
    set +u
    source .venv/bin/activate
    set -u
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
ft() { python -m fragtrust.cli "$@"; }
