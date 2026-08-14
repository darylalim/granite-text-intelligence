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

# Skip only when re-running is *provably* redundant. Two cheaper short-circuits
# were tried and both failed open, because each inferred "nothing to check" from
# a proxy rather than from the gate's actual inputs: an empty `git status
# --porcelain` skipped exactly the turns that had just committed (committing
# clears dirtiness without clearing risk) and a held index.lock silently
# disabled the gate outright, while `stop_hook_active` made it one-shot — it
# blocked once, then let the next Stop through without re-running anything.
#
# Hashing the inputs has neither shape. Committing does not alter file
# *content*, so the hash is unchanged and skipping is correct — that content
# really was verified. Nothing is recorded on a red gate, so it re-runs until
# green and verifies its own fix by construction. And it beats the tempting
# marker-file variant (PostToolUse writes "dirty", Stop reads it), which fails
# open on every edit that does not go through a tool hook: `sed -i`, `git
# checkout`, applying a patch.
#
# Every uncertainty runs the gate. Note an unreadable tree must NOT fall through
# to `shasum` of nothing: that is a stable, non-empty hash, so it could match a
# stored one. An empty file list is therefore a refusal, not an answer.
fingerprint() {
	list=$(git ls-files -co --exclude-standard 2>/dev/null) || return 1
	[ -n "$list" ] || return 1
	git ls-files -co --exclude-standard -z 2>/dev/null |
		xargs -0 shasum -a 256 2>/dev/null |
		shasum -a 256 2>/dev/null |
		cut -d' ' -f1
} </dev/null # hang-proof: xargs can run shasum with no file arguments

memo=''
git_dir=$(git rev-parse --git-dir 2>/dev/null) && memo="$git_dir/gate-ok"
before=$(fingerprint) || before=''

# CI never skips. A fresh checkout has no memo, so this is belt-and-braces —
# but the guarantee should hold by construction rather than by luck, since
# caching the workspace would restore a memo along with everything else.
if [ -z "${GITHUB_ACTIONS:-}" ] && [ -z "${GATE_FORCE:-}" ] &&
	[ -n "$memo" ] && [ -n "$before" ] &&
	[ "$before" = "$(cat "$memo" 2>/dev/null)" ]; then
	echo 'gate: no source file changed since the last green run — skipped (GATE_FORCE=1 to override)'
	exit 0
fi

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
	# Deliberately leaves any existing memo alone rather than clearing it: it
	# records a tree that genuinely passed, so reverting to that tree should
	# still skip. The current (failing) tree hashes differently and re-runs.
	printf '%s\n' "$failures" >&2
	exit 2
fi

# Recorded *after* the run, not before: every gate command goes through `uv run`,
# which silently re-locks, so uv.lock can differ by the time we get here. Storing
# the pre-run hash would then never match the tree the next Stop actually sees,
# and the memo would never once hit.
if [ -n "$memo" ]; then
	after=$(fingerprint) || after=''
	[ -z "$after" ] || printf '%s\n' "$after" >"$memo" 2>/dev/null || :
fi

echo 'gate: ruff check, ruff format --check, ty check, pytest — all passed'
