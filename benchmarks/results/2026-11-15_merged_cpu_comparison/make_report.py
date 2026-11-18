"""Regenerate measured tables without importing or running detector libraries."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = Path(__file__).resolve().parent
LABELS = {
    "napa": "NAPA (default, min_tag_px=22, split_touching=auto)",
    "napa-mtp16": "NAPA, min_tag_px=16",
    "napa-mtp32": "NAPA, min_tag_px=32",
    "wpilib-d2": "AprilTag, WPILib defaults",
    "apriltag3-d2": "AprilTag, decimate 2",
    "apriltag3-d1": "AprilTag, full resolution",
    "locus": "Locus standard",
    "locus-d2": "Locus, decimate 2",
    "opencv": "OpenCV ArUco",
    "opencv-subpix": "OpenCV ArUco, subpixel",
    "opencv-aruco3": "ArUco3 full",
    "opencv-aruco3-min32": "ArUco3, minimum ≈64 px",
    "opencv-aruco3-half16": "ArUco3, minimum ≈32 px",
    "aruco-nano": "ArUco Nano",
    "rapidtag": "RapidTag",
    "rapidtag-subpix": "RapidTag, subpixel",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numbers(row, field, cast=float):
    return [cast(v) for v in row[field].split(";") if v]


def percentiles(values, points=(50, 95, 100), scale=1):
    return (np.percentile(values, points) / scale).tolist() if values else [None] * len(points)


def fmt(value):
    return "—" if value is None else f"{value:.3f}"


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def summarize(rows):
    tags = sum(int(r["tags"]) for r in rows)
    found = sum(int(r["true_positive"]) for r in rows)
    return {
        "frames": len(rows),
        "tags": tags,
        "found": found,
        "recall": found / tags if tags else None,
        "false_ids": sum(int(r["false_id"]) for r in rows),
        "duplicates": sum(int(r["duplicate"]) for r in rows),
        "p50_p95_p99_ms": percentiles([n for r in rows for n in numbers(r, "samples_ns", int)], (50, 95, 99), 1e6),
        "corner_median_p95_max_px": percentiles([n for r in rows for n in numbers(r, "corner_errors_px")]),
    }


def summary_table(summary):
    return table(
        ["Detector", "p50 ms", "p95 ms", "p99 ms", "Found / tags", "Wrong IDs", "Duplicates"],
        [
            [LABELS[b], *map(fmt, s["p50_p95_p99_ms"]), f"{s['found']}/{s['tags']}", s["false_ids"], s["duplicates"]]
            for b, s in summary.items()
        ],
    )


def validate(measured, protocol, rows):
    cfg = measured["run_config"]
    backends = protocol["backends"]
    require(set(backends) == set(measured["backends"]) == set(LABELS), "Backend inventory differs")
    require(not measured["skipped_backends"], "Skipped backends")
    require(
        datetime.fromisoformat(protocol["recorded_before_measurement_utc"])
        <= datetime.fromisoformat(measured["started_at_utc"])
        < datetime.fromisoformat(measured["finished_at_utc"]),
        "Protocol/run timestamps out of order",
    )
    for b, settings in measured["backends"].items():
        if "threads" in settings:
            require(settings["threads"] == 1, f"Worker settings: {b}")
    require(not cfg["only_frames"] and not cfg.get("real_images"), "Filtered or real-image run")
    for key, expected in {
        "seed": protocol["seed"],
        "order_seed": protocol["order_seed"],
        "warmup": protocol["warmup_per_frame"],
        "repetitions": protocol["timed_repetitions"],
        "threads": protocol["threads"],
        "frames": protocol["main"]["frames_per_scenario_per_resolution"],
        "stress_frames": protocol["stress"]["frames_per_scenario_per_resolution"],
        "range_cases": protocol["range"]["frames_per_size"],
    }.items():
        require(cfg[key] == expected, f"Run/protocol mismatch: {key}")
    require(cfg["threads"] == 1 and protocol["cpus"] == [2], "Expected single worker on CPU2")
    require(measured["machine"]["affinity"] == protocol["cpus"], "Machine affinity mismatch")
    require(measured["threads_observed"], "Missing thread observations")
    for phase in measured["threads_observed"].values():
        require(bool(phase) and all(t["cpus"] == protocol["cpus"] for t in phase), "Thread affinity mismatch")
    require(protocol["source_commit"] and protocol["source_sha256"], "Missing source provenance")
    for name, digest in protocol["source_sha256"].items():
        require(sha256(ROOT / name) == digest, f"Source changed since measurement protocol: {name}")
    binary = ROOT / protocol["napa_binary_path"]
    if binary.exists():
        require(sha256(binary) == protocol["napa_binary_sha256"], "Binary hash mismatch")
    for b, minimum in (("napa", 22), ("napa-mtp16", 16), ("napa-mtp32", 32)):
        settings = measured["backends"][b]
        require(settings["commit"] == protocol["source_commit"], f"Source commit: {b}")
        require(settings["core_library_sha256"] == [protocol["napa_binary_sha256"]], f"Binary mismatch: {b}")
        require(settings["min_tag_px"] == minimum and settings["split_touching"] == "auto", f"Settings: {b}")
        require(settings["max_detections"] == 64, f"Output capacity: {b}")
    for b, settings in protocol.get("expected_backend_settings", {}).items():
        require(all(measured["backends"][b][k] == v for k, v in settings.items()), f"Settings mismatch: {b}")
    resolutions = protocol["resolutions"]
    actual_resolutions = ["x".join(re.findall(r"\d+", r)) for r in cfg["resolutions"]]
    require(actual_resolutions == resolutions, "Resolution mismatch")
    require(not set(cfg["scenarios"]) & set(cfg["stress"]), "Overlapping scopes")
    expected = {}
    for scope, scenes, count in (
        ("main", cfg["scenarios"], cfg["frames"]),
        ("stress", cfg["stress"], cfg["stress_frames"]),
    ):
        require(len(scenes) == protocol[scope]["scenarios"], f"Scenario count: {scope}")
        expected[scope] = {f"{s}-{res}-{i:03d}" for s in scenes for res in resolutions for i in range(count)}
    smallest = min(resolutions, key=lambda r: np.prod([int(x) for x in r.split("x")]))
    expected["range"] = {
        f"range-{smallest}-{i:03d}" for i in range(cfg["range_cases"] * len(protocol["range"]["sizes_px"]))
    }
    for scope, frames in expected.items():
        require(len(frames) == protocol[scope]["frames"], f"Protocol frame count: {scope}")
    all_frames = set.union(*expected.values())
    require(len(all_frames) == protocol["total_frames"], "Total frame count")
    grouped = defaultdict(dict)
    for r in rows:
        b, frame = r["backend"], r["frame"]
        require(b in backends and frame not in grouped[b], f"Unknown backend or duplicate row: {b}/{frame}")
        grouped[b][frame] = r
        require(r["stable_across_repetitions"] == "True", f"Unstable row: {b}/{frame}")
        for field in ("samples_ns", "cpu_ns"):
            samples = numbers(r, field, int)
            require(len(samples) == cfg["repetitions"] and min(samples) > 0, f"Invalid samples: {b}/{frame}")
        errors = numbers(r, "corner_errors_px")
        require(
            len(errors) == int(r["true_positive"]) and all(np.isfinite(x) and x >= 0 for x in errors),
            f"Invalid corner errors: {b}/{frame}",
        )
    require(len(rows) == len(backends) * len(all_frames), "Row count mismatch")
    for b in backends:
        require(set(grouped[b]) == all_frames, f"Incomplete coverage: {b}")
        for frame, r in grouped[b].items():
            require(
                all(
                    r[k] == grouped["napa"][frame][k]
                    for k in ("image_sha256", "tags", "expected_ids", "scenario", "resolution", "min_side_px")
                ),
                f"Shared frame/truth mismatch: {b}/{frame}",
            )
    max_truth = max(int(r["tags"]) for r in rows)
    max_napa = max(len(numbers(r, "detected_ids", int)) for r in rows if r["backend"].startswith("napa"))
    require(max_truth < 64 and max_napa < 64, "64-output cap may affect results")
    return expected, {
        "max_truth_per_frame": max_truth,
        "max_napa_detections": max_napa,
        "binary_verified_locally": binary.exists(),
    }


def historical_corpus_audit(rows):
    """Check pixel-hash prefixes and annotation identity against the original fixed run."""
    reference = ROOT / "benchmarks/results/2026-10-30_cpu_full_corpus/matrix.csv"
    current = {r["frame"]: r for r in rows if r["backend"] == "napa"}
    with reference.open(newline="") as source:
        original = {r["frame"]: r for r in csv.DictReader(source) if r["backend"] == "napa"}
    require(set(current) <= set(original), "Frames are outside the original fixed corpus")
    fields = ("image_sha256", "tags", "expected_ids", "scenario", "resolution")
    for frame, row in current.items():
        require(
            all(row[k] == original[frame][k] for k in fields),
            f"Fixed corpus pixel hash or annotation differs: {frame}",
        )
    return {
        "reference": reference.relative_to(ROOT).as_posix(),
        "reference_sha256": sha256(reference),
        "frames_matched": len(current),
        "pixel_hash_prefix_bits": 64,
        "fields_matched": list(fields),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    measured = json.loads((RESULTS / "matrix.json").read_text())
    protocol = json.loads((RESULTS / "protocol.json").read_text())
    csv_path = RESULTS / "matrix.csv"
    if not csv_path.exists():
        csv_path = RESULTS / "matrix.csv.gz"
    with gzip.open(csv_path, "rt", newline="") if csv_path.suffix == ".gz" else csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    expected, audit = validate(measured, protocol, rows)
    scopes = {scope: [r for r in rows if r["frame"] in frames] for scope, frames in expected.items()}
    summaries = {
        scope: {b: summarize([r for r in rs if r["backend"] == b]) for b in LABELS} for scope, rs in scopes.items()
    }
    paired = {}
    for scope, rs in scopes.items():
        medians = {(r["backend"], r["frame"]): statistics.median(numbers(r, "samples_ns", int)) for r in rs}
        paired[scope] = {
            b: statistics.median(medians["wpilib-d2", f] / medians[b, f] for f in expected[scope]) for b in LABELS
        }
    range_rows = []
    for b in LABELS:
        cells = []
        for size in protocol["range"]["sizes_px"]:
            rs = [r for r in scopes["range"] if r["backend"] == b and float(r["detail_side_px"]) == size]
            require(len(rs) == protocol["range"]["frames_per_size"], f"Range size coverage: {b}/{size}")
            cells.append(f"{sum(int(r['true_positive']) for r in rs)}/{sum(int(r['tags']) for r in rs)}")
        range_rows.append([LABELS[b], *cells])
    range_table = table(["Detector / black-square px", *map(str, protocol["range"]["sizes_px"])], range_rows)
    analysis = {
        "scopes": summaries,
        "median_paired_speedup_vs_wpilib": paired,
        "capacity_audit": audit,
        "historical_corpus_audit": historical_corpus_audit(rows),
        "files_sha256": {p.name: sha256(p) for p in (RESULTS / "matrix.json", RESULTS / "protocol.json", csv_path)},
    }
    date = measured["started_at_utc"][:10]
    machine = measured["machine"]
    hardware = machine["cpu_model"]
    method = (
        f"{date}: {len(expected['main']):,} main frames, {len(expected['stress']):,} stress frames and "
        f"{len(expected['range']):,} range frames, shared by all {len(LABELS)} configurations. "
        f"Each frame has {protocol['warmup_per_frame']} warm-up and {protocol['timed_repetitions']} timed calls "
        f"per detector, seed/order seed {protocol['seed']}/{protocol['order_seed']}, single-threaded on CPU2 of "
        f"{hardware}. "
        "Timing includes detector calls and adapter conversion, excluding construction, rendering and scoring. "
        "Latency percentiles pool timed calls; quality counts each frame once. Speedup is the median of paired "
        "per-frame median WPILib time divided by detector time. Found counts match IDs without a geometric gate. "
        "Corner order is aligned over cyclic shifts and reflections before measuring error. "
        "Corner error is the worst corner per matched tag; "
        "its distribution conditions on detection. NAPA's current default is min_tag_px=22, split_touching=auto."
    )
    limits = (
        "These synthetic images and single-host timings do not establish real-camera accuracy or a universal "
        "winner. This run does not measure multicore throughput or isolated RSS. CPU frequency varied and package "
        "temperatures reached 97°C; no throttle counters were captured. See the [measurement audit](audit.md) "
        "for monitor observations and limits."
    )
    report = [f"# CPU comparison — {date}", method]
    for scope, summary in summaries.items():
        report += [
            f"## {scope.title()}: {len(expected[scope]):,} frames",
            summary_table(summary),
            table(
                ["Detector", "Median paired speedup vs WPILib", "Corner median / p95 / max px"],
                [
                    [LABELS[b], f"{paired[scope][b]:.2f}×", " / ".join(map(fmt, s["corner_median_p95_max_px"]))]
                    for b, s in summary.items()
                ],
            ),
        ]
    report += ["## Range by size", range_table, "## Per-scene main results"]
    for resolution in protocol["resolutions"]:
        for scene in measured["run_config"]["scenarios"]:
            report += [
                f"### {scene}, {resolution}",
                summary_table(
                    {
                        b: summarize(
                            [
                                r
                                for r in scopes["main"]
                                if r["backend"] == b and r["scenario"] == scene and r["resolution"] == resolution
                            ]
                        )
                        for b in LABELS
                    }
                ),
            ]
    report += [
        "## Provenance and limitations",
        limits,
        f"Platform: `{machine['platform']}`. Full machine metadata and effective backend versions/settings are in "
        "matrix.json. Frame pixel-hash prefixes and annotations were also checked against the original fixed run; "
        "the hashes cover 64-bit SHA256 prefixes, not full pixel digests.",
        f"Source commit: `{protocol['source_commit']}`. NAPA binary SHA256: `{protocol['napa_binary_sha256']}`. "
        f"Maximum truth/NAPA detections: {audit['max_truth_per_frame']}/{audit['max_napa_detections']}, below 64. "
        "Source hashes and recorded NAPA settings were checked against protocol.json; peer binary hashes are not "
        "a complete inventory of loaded native code. "
        f"Binary available and hash-verified locally: {audit['binary_verified_locally']}.",
        "[Protocol](protocol.json), [raw metadata/settings](matrix.json), "
        f"[per-frame measurements]({csv_path.name}), [analysis](analysis.json), "
        "[commands](commands.md), [adapter validation](validate.json), [monitor](monitor.tsv).",
    ]
    (RESULTS / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    (RESULTS / "report.md").write_text("\n\n".join(report) + "\n")
    print(f"Regenerated {RESULTS / 'report.md'}")


if __name__ == "__main__":
    main()
