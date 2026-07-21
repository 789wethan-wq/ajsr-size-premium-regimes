# expected_outputs/ is not yet populated

This directory is intentionally empty as of this commit. The pipeline code
in `src/` (including tonight's fixes to `bootstrap_inference.py`'s seeding
and `robustnesstests.py`'s three-state multi-start search) has NOT yet had
its outputs regenerated and verified into this directory.

Do not treat an empty `expected_outputs/` as "verification passed" --
`verify_against_expected.py` will currently report "0 files checked, 0
with real issues" against an empty directory, which is a silent pass, not
a real one. Do not rely on that output until this file is gone and real
CSVs/PNGs are here instead.

**What's pending:** two fully independent fresh-clone runs of this exact
repo's `run_pipeline.sh`, diffed directly against each other (exact match
or <1e-8 = pass) before anything is written here -- the same discipline
used to find and fix the `bootstrap_inference.py` seeding bug. That
verification is in progress as of this commit; see CHANGELOG.md.

This file will be deleted in the follow-up commit that adds the real
`expected_outputs/` contents.
