# Measuring on your own footage

Synthetic benchmarks can't stand in for your camera, lighting, and tags. `napa-benchmark` scores NAPA on a folder of
your own labeled images, and needs the `headless` extra.

Copy `benchmarks/corpus.example.json` and list each image with the tag IDs it contains. Add a `sha256` to any frame to
freeze it, so a changed image fails the run instead of quietly changing the results.

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

Paths are relative to the manifest and must stay inside its folder. Then run:

```bash
napa-benchmark my-corpus/manifest.json \
  --repetitions 5 --warmup-passes 1 \
  --min-recall 0.99 --min-precision 1.0 --max-p95-total-ms 30 \
  --output report.json --pretty
```

The report gives recall, precision, missed and unexpected IDs (with up to 25 examples), and p50/p95/p99 times for
loading, detection, scoring, and the total. It exits with `1` if a `--min-*` or `--max-*` gate fails, and `2` on bad
input, so you can run it in CI to catch regressions on footage you trust.

To compare NAPA with AprilTag on synthetic scenes, see [the benchmark guide](../benchmarks/README.md).
