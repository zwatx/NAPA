# Measuring on your own footage

Synthetic benchmarks cannot stand in for your camera, your lighting, and your tags. `napa-benchmark` scores NAPA on
a folder of images you label yourself. It needs the `headless` extra.

Copy `benchmarks/corpus.example.json` and list each image with the tag IDs it contains:

```json
{
  "schema_version": 1,
  "name": "my-camera",
  "frames": [
    {"path": "frames/0001.png", "expected_tag_ids": [1, 7], "sha256": "<64 hex characters>"},
    {"path": "frames/0002.png", "expected_tag_ids": []}
  ]
}
```

Paths are relative to the manifest and must stay inside its folder. The `sha256` field is optional. Add it to freeze
a frame, so a changed image fails the run instead of quietly changing the results. Include frames with no tags.
They are the ones that catch false positives.

Then run:

```bash
napa-benchmark my-corpus/manifest.json \
  --repetitions 5 --warmup-passes 1 \
  --min-recall 0.99 --min-precision 1.0 --max-p95-total-ms 30 \
  --output report.json --pretty
```

The report gives recall, precision, missed and unexpected IDs (with up to 25 examples of each), and p50, p95, and
p99 times for loading, detection, scoring, and the total.

Exit codes make it usable in CI. It exits 1 when a `--min-*` or `--max-*` gate fails, and 2 on bad input. Pin a
corpus of footage you trust, gate on it, and a detector upgrade that loses tags on your field fails the build.

From Python, `napa_vision.benchmark.run_replay_benchmark` does the same work and accepts any detector that follows
the `FiducialDetector` protocol. That is the way to run the same corpus through NAPA and through another detector
you wrap yourself.

To compare NAPA with other detectors on synthetic scenes with exact corner truth, see the
[benchmark guide](../benchmarks/README.md).
