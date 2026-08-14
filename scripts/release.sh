#!/bin/sh
# Tag and publish a GitHub release when pyproject.toml's version has been bumped.
#
# Invoked by the `release` job in .github/workflows/ci.yml, which reaches it only
# on a push to main whose `check` job passed: a tag never points at a red build,
# and a pull request that bumps the version has to land before it can release.
#
# The trigger is "no tag exists for the current version", not "pyproject.toml
# changed in this push". A diff-based trigger has to answer "changed since what?"
# and every answer is wrong somewhere — github.event.before is all-zeros on a
# branch's first push, a squash-merge rewrites the range, and a re-run of a failed
# job sees no diff at all. Asking the remote whether v<version> is already tagged
# is the same question with no state to get wrong, so re-runs, force-pushes and
# manual invocations all converge on the same outcome.
#
# Every uncertainty is a refusal. An unreadable version, an inconclusive tag
# probe, a commit that is not on main — each stops the run rather than guessing,
# because the artifact is a public tag that people install.
#
# Exits 0 both when it releases and when there is nothing to release; non-zero
# only on a genuine failure. Deliberately plain exit 1, not gate.sh's exit 2 —
# that 2 is a Claude Code hook contract, and this script is not a hook.

set -u

# Anchor to the repo root: `uv version` reads ./pyproject.toml and gh infers the
# repository from the working directory's git remote (unless GH_REPO is set, as
# the workflow does), so a caller sitting in a subdirectory would resolve neither.
# Same fallback as gate.sh for the cases where git is unavailable.
root=$(git rev-parse --show-toplevel 2>/dev/null) || root=''
[ -n "$root" ] || root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd -- "$root" || {
	echo "release: cannot enter repo root ($root)" >&2
	exit 1
}

for tool in gh uv; do
	command -v "$tool" >/dev/null 2>&1 || {
		echo "release: $tool is required but is not installed" >&2
		exit 1
	}
done

# `uv version --short` reads [project].version out of pyproject.toml with a real
# TOML parser, so there is no hand-rolled regex to mis-read a version. `--short`
# is load-bearing: bare `uv version` prints "<name> <version>", and before uv 0.7
# it reported uv's *own* version entirely (now `uv self version`). uv's stderr is
# deliberately not captured, so its own diagnostic reaches the log unaltered.
version=$(uv version --short) || {
	echo 'release: could not read a version from pyproject.toml (see uv above)' >&2
	exit 1
}

# Rejects every shape `uv version` prints that is not a bare version — "uv 0.6.14"
# and "granite-text-intelligence 0.1.0" fail the leading-digit test — and anything
# carrying a character that has no business in a tag. Checking only the first
# character would admit "0.1.0 (from pyproject.toml)" and any future stdout
# decoration that happens to lead with a digit, and tag it verbatim.
case "$version" in
'' | [!0-9]* | *[!0-9A-Za-z.+-]*)
	echo "release: refusing to tag an implausible version ('$version')" >&2
	exit 1
	;;
esac

tag="v$version"

# Ask the remote, not the local clone: the workflow checks out at depth 1 with no
# tags fetched, so `git tag -l` would find nothing and re-release on every push.
# `git/ref/tags/<tag>` (singular `ref`) is an exact lookup — the plural
# `git/refs/tags/<tag>` form prefix-matches, and would report the existing v0.1.0
# as a tag for version 0.1.
probe=$(gh api "repos/{owner}/{repo}/git/ref/tags/$tag" 2>&1)
probed=$?

if [ "$probed" -eq 0 ]; then
	echo "release: $tag is already tagged — nothing to release"
	exit 0
fi

# Only a definite 404 means "not released yet". Rate limiting, an expired token
# and a 5xx all fail the same way, and treating those as "absent" would try to
# release a version that already shipped. Refuse instead of guessing.
case "$probe" in
*'(HTTP 404)'*) ;;
*)
	echo "release: could not determine whether $tag exists — refusing to release" >&2
	printf '%s\n' "$probe" >&2
	exit 1
	;;
esac

# Tag the commit the workflow actually validated, not whatever the default branch
# has drifted to since the run started. gh reads an empty --target as "use the
# default branch", so an unresolvable commit has to be an error rather than a
# fallback.
target=${GITHUB_SHA:-}
[ -n "$target" ] || target=$(git rev-parse HEAD 2>/dev/null) || {
	echo 'release: cannot resolve a commit to tag' >&2
	exit 1
}

# Outside Actions there is no workflow `if:` confining this to main, and HEAD is
# whatever branch the caller happens to be on — so a hand-run on a feature branch
# would publish a public tag pointing at an unreviewed commit. In Actions the
# workflow has already made this guarantee, and origin/main may not even be
# fetched at depth 1.
if [ "${GITHUB_ACTIONS:-}" != "true" ]; then
	git merge-base --is-ancestor "$target" origin/main 2>/dev/null || {
		echo "release: $target is not an ancestor of origin/main" >&2
		echo 'release: releases are cut from main; fetch origin, or let CI do it' >&2
		exit 1
	}
fi

# A draft release carries no git tag, so the probe above cannot see one — and
# GitHub accepts a *second* draft for the same tag_name precisely because there is
# no tag to collide with. A failed publish would therefore leave a draft that the
# next run neither sees nor replaces, and `gh release edit` would then resolve
# whichever of the two it happened to find first. Adopt the existing draft instead
# of creating a rival, and retarget it so the tag still lands on the commit *this*
# run validated.
drafts=$(gh release list --limit 100 --json tagName,isDraft \
	--jq '.[] | select(.isDraft) | .tagName') || {
	echo 'release: could not list existing releases — refusing to release' >&2
	exit 1
}

publish_failed() {
	echo "release: $tag is drafted but could not be published — publish or delete the draft, then re-run" >&2
	exit 1
}

# Draft first, publish second, on purpose. Because a draft holds no tag until it
# is published, the tag — the very thing the probe above reads — only comes into
# existence at the last moment. Anything failing before then leaves no tag to be
# mistaken for a finished release: a single `gh release create` that died after
# tagging would leave a tag with no release, which the probe would read as
# "already released" and skip forever, silently.
#
# --title is passed explicitly even though --generate-notes would synthesize one:
# every release so far is titled exactly after its tag.
if printf '%s\n' "$drafts" | grep -qxF "$tag"; then
	echo "release: adopting the $tag draft an earlier run left behind"
	gh release edit "$tag" --target "$target" --draft=false || publish_failed
else
	echo "release: drafting $tag at $target"
	gh release create "$tag" \
		--target "$target" \
		--title "$tag" \
		--generate-notes \
		--draft || {
		echo "release: failed to draft $tag" >&2
		exit 1
	}

	gh release edit "$tag" --draft=false || publish_failed
fi

echo "release: published $tag"
