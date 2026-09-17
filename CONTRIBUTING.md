# Contributing to weqmmm

Start with [the undergraduate handoff](docs/UNDERGRAD_HANDOFF.md).
Every change should record what changed, why, the exact command used, its result,
and any remaining limitation in the relevant documentation.

Before opening a pull request:

1. Activate the documented `weqmmm` environment.
2. Run `python scripts/check_prereqs.py`.
3. Run the focused tests for the code you changed, then the non-GPU regression.
4. Preserve failed logs and diagnostic geometry; do not hide an SCF failure.

Do not add SCF retries, silent fallbacks, adaptive QM selection during MD, or
changes to wepy/PartQMMM under `external/`. New QM engines belong behind the
provider protocol in `weqmmm/engines/` and must report QM and MM reaction forces.

The repository currently carries the MIT License in `LICENSE`. Confirm that
the copyright holder and the inclusion of the example research inputs are
acceptable before publishing a public release.
