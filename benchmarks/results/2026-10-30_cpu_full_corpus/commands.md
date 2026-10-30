# Fixed full-corpus CPU comparison

The protocol in [protocol.json](protocol.json) was recorded before timing. The renderer, scoring, adapters and
native source were frozen at `f7a7adfe27ca701aee75561be924f64b83bdf442`; source and binary hashes are recorded there.
The detector algorithm is unchanged from the optimized source already published in PR #14.

The single run uses the existing `cpu_comparison.py latency` command with its ordinary `napa` (minimum 32 px) and
`napa-mtp16` backend options alongside all 13 competitor configurations. Every backend receives exactly the same
4,500 frames. No frame selection depends on NAPA's prior losses or any detector's outputs.

- Main corpus: all 20 scenarios × 50 independent renders × 3 resolutions = 3,000 frames.
- Stress: all four stress scenarios × 50 × 3 resolutions = 600 frames, reported separately.
- Range: all 18 existing sizes × 50 cases = 900 frames at 640×480, reported separately.
- Seed and backend-order seed: 2468. One warm-up plus ten timed calls per image/backend, shuffled backend order.
- Affinity: CPU2, a verified performance core of this i7-12700H (SMT sibling CPU3). Every internal pool is set to one.
- Both NAPA options retain their normal public-adapter settings and 64-output capacity. Frame truth counts and
  detection counts must be audited for any capacity saturation before quality claims are made.

Detector construction, rendering, scoring and process startup are excluded from detector-call timing. Required
input copies, internal processing and Python result conversion remain included. CPU time is process CPU time around
each call. This interleaved run does not provide backend-isolated RSS measurements. Earlier isolated resource results
remain separate, with their original data, date and 96-frame scope; they are not measurements on this new full set.

## Reproduce

Use the isolated third-party build/venv/shim setup from
[the original CPU comparison](../2026-10-28_cpu_comparison/commands.md), but do not execute its old measurement scripts
or overwrite historical result directories. The additional dependencies used here are:

```bash
work/venv/bin/python -m pip install robotpy-apriltag==2026.2.2 ruff==0.17.1
```

From the repository root, on the measured host:

```bash
NAPA_FULL_OUTPUT_DIR=work/full-corpus/reproduction \
  bash benchmarks/results/2026-10-30_cpu_full_corpus/run.sh
```

`run.sh` checks the frozen source and rebuilt binary hashes before timing and refuses to overwrite a completed
matrix. The commands hard-code the measured host's P-core (CPU2) and should be adapted only after inspecting another host's
topology. Debug paths/compiler differences can change a rebuilt binary hash on another checkout; do not silently
replace the recorded manifest or relabel another binary as the measured build.

`validate.json` records adapter checks on known tags, rotations, geometry and blank frames. RapidTag finds 7/11
known tags while the other configurations find 11/11; all have zero wrong IDs, duplicates and blank detections in
that validation. These misses remain a quality tradeoff, not an excuse to drop RapidTag from the comparison.
`smoke.json`/`.csv` are a one-frame, two-repeat preflight only and are excluded from final tables.

`monitor.tsv` records temperature, frequency, CPU load and sibling activity throughout the full run. Existing
background applications are left running; no system-wide tuning, pool changes outside the benchmark process,
GPU implementation or GPU migration is used. The full matrix retains outliers and every preselected frame.

## Regenerate the report and public tables

After the fixed run, regenerate only the saved-data report (no detection or timing):

```bash
work/venv/bin/python benchmarks/results/2026-10-30_cpu_full_corpus/make_report.py
```

Add `--write-docs` to replace the comparison sections in `README.md`, `benchmarks/README.md` and `docs/pypi.md`.
The package-description source uses the same comparison text with absolute repository links; this does not
publish a package version or change an existing PyPI release. `analysis.json` records main, stress and range
statistics separately, paired ratios against both ordinary NAPA backend options, common-tag corner errors and
hashes of the raw inputs. The measured CSV and JSON are immutable; regeneration does not overwrite them.
