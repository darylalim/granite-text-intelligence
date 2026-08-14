#!/bin/sh
# The project's quality gate, in one place.
#
# Invoked by CI (.github/workflows/ci.yml) and by the Stop hook
# (.claude/settings.json) so the two cannot drift; TestHooksConfig asserts that
# both still call it and that it still runs all four documented gates.
#
# Exits 0 on pass, 2 on failure with every gate's output on stderr. 2 rather
# than 1 is deliberate: Claude Code treats *only* exit 2 from a hook as
# blocking — exit 1 is a non-blocking error that lets the turn end on a red
# gate — while any non-zero fails a CI job, so one convention serves both
# callers. stderr for the same reason: it is the stream fed back on a block,
# whereas ruff and pytest print their diagnostics to stdout.

set -u

# Anchor to the repo root. `ruff check .` and `pytest` resolve from the working
# directory, so a caller sitting in a subdirectory would otherwise check a
# narrower tree than it believes it is checking. Fall back to this script's own
# parent when git is unavailable (tarball export, broken index) rather than
# silently gating less.
root=$(git rev-parse --show-toplevel 2>/dev/null) || root=''
[ -n "$root" ] || root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd -- "$root" || {
	echo "gate: cannot enter repo root ($root)" >&2
	exit 2
}

failures=''

# Every gate runs even after an earlier one fails, so a single block reports
# everything that is wrong rather than only the first thing. The caller gets a
# bounded number of retries — Claude Code stops honoring a Stop hook after 8
# consecutive blocks — so spending them one diagnostic at a time is wasteful.
run_gate() {
	name=$1
	shift
	output=$("$@" 2>&1)
	code=$?
	[ "$code" -eq 0 ] && return 0
	# An empty capture means the process died without writing anything: an OOM
	# kill, a hook timeout, a half-created .venv. Say that, rather than failing
	# with a blank message the caller cannot act on.
	[ -n "$output" ] || output="(no output — exited $code without writing to stdout or stderr)"
	failures=$(printf '%s\n--- %s failed (exit %s) ---\n%s' "$failures" "$name" "$code" "$output")
}

run_gate 'ruff check' uv run ruff check .
run_gate 'ruff format --check' uv run ruff format --check .
run_gate 'ty check' uv run ty check
run_gate 'pytest' uv run pytest -q

if [ -n "$failures" ]; then
	printf '%s\n' "$failures" >&2
	exit 2
fi

echo 'gate: ruff check, ruff format --check, ty check, pytest — all passed'
