import csv
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from benchmarks.cpu_comparison import common_corners  # noqa: E402

R = Path(os.environ.get("NAPA_BENCH_OUTPUT_DIR", ROOT / "benchmarks/results/2026-10-29_cpu_optimization"))


def read(name):
    with (R / f"{name}.csv").open() as file:
        return list(csv.DictReader(file))


lat = read("final_latency")
by = defaultdict(list)
for row in lat:
    by[row["backend"]].append(row)
statistics = {}
for b, rows in by.items():
    samples = [int(v) / 1e6 for x in rows for v in x["samples_ns"].split(";")]
    cpu = [int(v) / 1e6 for x in rows for v in x["cpu_ns"].split(";")]
    statistics[b] = dict(
        p50_p95_p99_ms=np.percentile(samples, [50, 95, 99]).tolist(),
        cpu_mean_ms=float(np.mean(cpu)),
        found=sum(int(x["true_positive"]) for x in rows),
        truth=sum(int(x["tags"]) for x in rows),
        false_ids=sum(int(x["false_id"]) for x in rows),
        duplicates=sum(int(x["duplicate"]) for x in rows),
    )
resources = {}
origins = {}
for prefix in ("resources", "final_resources"):
    for p in sorted(R.glob(f"{prefix}_*t_*.json")):
        j = json.loads(p.read_text())
        assert all("error" not in x and "skip_reason" not in x for x in j["results"]), p
        for x in j["results"]:
            key = (
                j["run_config"]["threads"],
                str(x["config"].get("min_tag_px", "")),
                x["backend"],
                p.stem.rsplit("_", 1)[1],
            )
            resources.setdefault(key, {})[x["round"]] = x
            origins[key] = p.name
resource_rows = []
for key, rounds in resources.items():
    ts, _, b, res = key
    xs = list(rounds.values())
    resource_rows.append(
        dict(
            threads=ts,
            backend=b,
            resolution=res,
            source=origins[key],
            rounds=len(xs),
            wall_ms=st.mean(x["wall_ms_per_frame"] for x in xs),
            cpu_ms=st.mean(x["cpu_ms_per_frame"] for x in xs),
            p50_p95_p99_ms=[st.mean(x["latency_ms"][k] for x in xs) for k in ("p50", "p95", "p99")],
            workspace_mib=st.mean((x["rss_kib"]["peak"] - x["rss_kib"]["after_construct"]) / 1024 for x in xs),
            backend_total_mib=st.mean((x["rss_kib"]["peak"] - x["rss_kib"]["before_backend"]) / 1024 for x in xs),
            busy_cpus=st.mean(x["mean_busy_cpus"] for x in xs),
        )
    )
qname = "final_quality"
assert (R / "final_quality.csv").exists(), "final quality replay is required"
quality = read(qname)
qgroups = defaultdict(list)
for x in quality:
    qgroups[x["backend"]].append(x)
old = ROOT / "benchmarks/results/2026-10-28_cpu_comparison"
corpus = list(csv.DictReader((old / "single_thread_latency.csv").open()))
ranges = list(csv.DictReader((old / "range_sweep_latency.csv").open()))
reference = [x for x in corpus if x["scenario"] != "range"] + ranges
quality_pairs = common_corners(reference + quality, "candidate-mtp16")
counts = {
    b: dict(
        found=sum(int(x["true_positive"]) for x in xs),
        truth=sum(int(x["tags"]) for x in xs),
        false_ids=sum(int(x["false_id"]) for x in xs),
        duplicates=sum(int(x["duplicate"]) for x in xs),
    )
    for b, xs in qgroups.items()
}
ref_baseline = {x["frame"]: x for x in read("quality") if x["backend"] == "baseline-mtp16"}
fields = ("tags", "expected_ids", "tp_ids", "false_id", "duplicate", "corner_errors_px", "alignments")
assert all(
    all(x[k] == ref_baseline[x["frame"]][k] for k in fields)
    for x in corpus
    if x["backend"] == "napa-mtp16" and x["scenario"] != "range"
)
assert all(x["image_sha256"] == ref_baseline[x["frame"]]["image_sha256"] for x in ranges)
ratios = []
d = {b: {x["frame"]: x for x in xs} for b, xs in by.items()}
for b in by:
    if b.startswith("candidate"):
        continue
    values = [float(x["median_ms"]) / float(d["candidate-mtp16"][f]["median_ms"]) for f, x in d[b].items()]
    ratios.append(
        dict(
            backend=b,
            median=float(np.median(values)),
            minimum=min(values),
            napa_slower_frames=sum(v < 1 for v in values),
        )
    )
manifest = json.loads((R / "source_manifest.json").read_text())
measured = json.loads((R / "final_latency.json").read_text())
measured_hash = measured["backends"]["candidate-mtp16"]["sha256"]
assert measured_hash == manifest["candidate_binary_sha256"]
retime = read("outlier_retime")
retime_groups = defaultdict(dict)
for row in retime:
    retime_groups[row["backend"]][row["frame"]] = row
retime_stats = {}
for backend in ("aruco-nano", "locus-d2"):
    values = [
        float(row["median_ms"]) / float(retime_groups["candidate-mtp16"][name]["median_ms"])
        for name, row in retime_groups[backend].items()
    ]
    retime_stats[backend] = {"losses": sum(v < 1 for v in values), "minimum": min(values)}
retime_repetitions = json.loads((R / "outlier_retime.json").read_text())["config"]["repetitions"]
retime_frames = len(retime_groups["candidate-mtp16"])
mc_tail = {
    x["backend"]: x["p50_p95_p99_ms"][2]
    for x in resource_rows
    if int(x["threads"]) == 6 and x["resolution"] == "1920x1080"
}
real_name = "final_real_agreement.json" if (R / "final_real_agreement.json").exists() else "real_agreement.json"
real = json.loads((R / real_name).read_text())["backends"]
real_ours = real["napa-mtp16"]
real_times = "/".join(f"{pair['median_ms']:.2f}" for pair in real_ours["pairs"])
tests_text = (R / "tests.txt").read_text()
assert " failed" not in tests_text and " error" not in tests_text
passed = int(re.search(r"(\d+) passed", tests_text)[1])
skipped = re.search(r"(\d+) skipped", tests_text)
test_summary = f"{passed} tests pass" + (
    f", {skipped[1]} skipped" if skipped else " with all optional backends installed"
)
sanitizer_calls = int(re.search(r"(\d+) native calls passed", (R / "sanitizers.txt").read_text())[1])
assert "All checks passed!" in (R / "lint.txt").read_text()
latency_win = all(
    all(a < b for a, b in zip(statistics["candidate-mtp16"]["p50_p95_p99_ms"], value["p50_p95_p99_ms"], strict=True))
    for backend, value in statistics.items()
    if not backend.startswith("candidate")
)
resource_win = True
for mode in (1, 6):
    for resolution in ("640x480", "1280x720", "1920x1080"):
        cohort = [x for x in resource_rows if int(x["threads"]) == mode and x["resolution"] == resolution]
        ours = next(x for x in cohort if x["backend"] == "napa-mtp16")
        resource_win &= all(
            all(ours[k] < peer[k] for k in ("wall_ms", "cpu_ms", "workspace_mib"))
            for peer in cohort
            if peer["backend"] not in ("napa", "napa-mtp16")
        )
headline = (
    "NAPA with `min_tag_px=16` has lower single-core p50/p95/p99 and lower mean wall time, "
    "CPU time and working memory than every tested competitor configuration on these workloads."
    if latency_win and resource_win
    else "This replay has mixed latency/resource results; inspect the per-configuration tables before claiming a win."
)

analysis = dict(
    latency=statistics,
    resources=resource_rows,
    quality=counts,
    quality_file=qname + ".csv",
    paired_ratios=ratios,
    common_corners=quality_pairs,
    prior_quality_audit=dict(corpus_baseline_fields_matching=1080, range_frames_hash_matching=540),
    source_sha256=manifest["candidate_source_sha256"],
    binary_sha256=measured_hash,
)
(R / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
overhead = json.loads((R / "adapter_overhead.json").read_text())["frames"]
overhead_us = [(x["p50_ms"]["public_adapter"] - x["p50_ms"]["direct_ctypes"]) * 1000 for x in overhead]
lines = [
    "# CPU optimization comparison — 2026-10-29",
    "",
    headline,
    ("It does **not** win every image, every multicore tail statistic, or every corner."),
    ("These are synthetic benchmark results on one i7-12700H, not proof of general real-camera accuracy."),
    "",
    "## Changes",
    "",
    (
        "- Preserve tiny borders on the original stride-one image; suppress noise with a "
        "higher local floor and recover low contrast on an averaged stride-two grid."
    ),
    (
        "- Keep overlapping proposal bands for large low-contrast tags; retain narrow tilted "
        "and small blurred tags on finer grids."
    ),
    (
        "- Reject implausible nonterminal proposals with a cheap border screen; preserve "
        "terminal-crop behavior and final decoding/refinement."
    ),
    ("- Pack component metadata by dense component index, preserving raster order and reducing sparse buffer touches."),
    (
        "- Defaults remain `min_tag_px=32`. Use `NapaDetector(min_tag_px=16)` for the "
        "high-recall comparison below. Corner refinement remains enabled; error correction is "
        "unchanged."
    ),
    "",
    "## Scope and timing",
    "",
    (
        "- Final latency: 149 cached diagnostic frames, all 20 scenarios and four stress cases"
        " at 640×480, 1280×720 and 1920×1080; one warm-up plus three shuffled timed calls per "
        "frame/backend."
    ),
    (
        "- The diagnostic set deliberately includes historical losses to Nano/Locus. A second "
        "render seed uses the same selection rule (149 frames, seven repetitions); it is "
        "corroboration, not an unbiased camera sample."
    ),
    (
        "- Full quality replay: 900 corpus + 180 stress + 540 range frames (18 sizes × 30), "
        "3,486 tags. Maximum placed tags in a frame is 33; the resource adapter’s 64-output "
        "limit is nonbinding."
    ),
    (
        "- Resources: identical saved 96-frame mixed sets per resolution; isolated processes, "
        "import/construction/warm-up/timed phases separated, one warm-up pass and two timed "
        "passes × two shuffled rounds. One allowed P-core (CPU2), then six physical P-cores "
        "(0,2,4,6,8,10). NAPA still uses one execution thread per frame."
    ),
    (
        "- Detectors are constructed once. Timing includes required internal processing, input"
        " copies and Python result conversion, but excludes rendering, scoring and process "
        "startup. NumPy/OpenBLAS is limited to one thread; all other pools and actual "
        "affinities are recorded."
    ),
    (
        "- Existing grayscale renderer/scoring and tag36h11 are unchanged. AprilTag/Locus "
        "origins are normalized by −0.5px. Geometric corner alignment and canonical ordering "
        "remain separate."
    ),
    (
        "- External Chrome load and thermal/frequency drift were left visible in both monitor "
        "TSVs. First/closest/worst frames were retimed separately, not discarded."
    ),
    "",
    (
        "Configuration legend: Locus d2 is the standard profile with decimation two. "
        "OpenCV/RapidTag defaults do not refine corners; SUBPIX variants are separate."
    ),
    (
        "ArUco3-min32 is a historical name: half-scale detection makes its effective minimum "
        "about64px; half16 is about32px. Full-scale ArUco3 performs no refinement at its "
        "default ratio despite forcing the SUBPIX flag."
    ),
    (
        "Pinned versions: upstream AprilTag3.4.5/b7c0ebe; WPILib2026.2.2; Locus0.10.0; OpenCV "
        "Python5.0.0.93; Nano c99989b with OpenCV4.14.0; RapidTag0.1.12/fad141c. Exact build "
        "flags/effective settings/actual threads are in each JSON and the linked original "
        "build commands."
    ),
    "",
    "## Final diagnostic latency",
    "",
    (
        "Pooled detector-call percentiles, milliseconds. The 149-frame mixture is deliberately"
        " difficult and unbalanced across resolutions; compare the resource tables for an "
        "equal scenario mixture."
    ),
    "",
    "| Configuration | p50 | p95 | p99 | Tags found / 319 | Wrong IDs |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for b, v in statistics.items():
    p = v["p50_p95_p99_ms"]
    lines.append(f"| {b} | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {v['found']} | {v['false_ids']} |")
lines += [
    "",
    ("Per-frame median time ratios (competitor ÷ NAPA16), on identical images. Values below one are real NAPA losses."),
    "",
    "| Configuration | Median ratio | Smallest ratio | NAPA slower frames / 149 |",
    "| --- | ---: | ---: | ---: |",
]
for x in ratios:
    lines.append(f"| {x['backend']} | {x['median']:.2f}× | {x['minimum']:.3f}× | {x['napa_slower_frames']} |")
lines += [
    "",
    (
        f"The {retime_frames}-frame, {retime_repetitions}-repeat retime finds Nano faster on "
        f"{retime_stats['aruco-nano']['losses']} selected frames and Locus d2 on {retime_stats['locus-d2']['losses']}. "
        f"Smallest repeated ratios are {retime_stats['aruco-nano']['minimum']:.3f}× and "
        f"{retime_stats['locus-d2']['minimum']:.3f}× respectively. Losses remain visible."
    ),
    "",
    "## CPU and working memory",
    "",
    (
        "Means of two process rounds. Working memory is peak RSS over the constructed "
        "detector, including warm-up and timed detection; image arrays were loaded before the "
        "baseline. Memory values average the two process peaks. RSS is approximate and includes allocator retention."
    ),
    (
        "NAPA/closest/WPILib entries use final packed-core runs; unchanged slower backends "
        "retain the earlier isolated measurements. Every row records its source JSON in "
        "`analysis.json`."
    ),
    "",
    (
        "| Cores allowed | Resolution | NAPA16 wall / CPU ms | Closest competitor mean wall | "
        "NAPA16 working MiB | Smallest peer working MiB |"
    ),
    "| --- | --- | ---: | --- | ---: | --- |",
]
for mode in (1, 6):
    for res in ("640x480", "1280x720", "1920x1080"):
        xs = [x for x in resource_rows if int(x["threads"]) == mode and x["resolution"] == res]
        ours = next(x for x in xs if x["backend"] == "napa-mtp16")
        peers = [x for x in xs if x["backend"] not in ("napa", "napa-mtp16")]
        speed = min(peers, key=lambda x: x["wall_ms"])
        memory = min(peers, key=lambda x: x["workspace_mib"])
        lines.append(
            f"| {mode} | {res} | {ours['wall_ms']:.3f} / {ours['cpu_ms']:.3f} "
            f"| {speed['backend']} {speed['wall_ms']:.3f} ms "
            f"| {ours['workspace_mib']:.2f} | {memory['backend']} {memory['workspace_mib']:.2f} |"
        )
lines += [
    "",
    (
        f"At 1080p, mean-of-round multicore p99 is {mc_tail['napa-mtp16']:.3f}ms for NAPA16, "
        f"{mc_tail['locus-d2']:.3f}ms for Locus d2 and {mc_tail['opencv-aruco3-min32']:.3f}ms for "
        "ArUco3 half-scale/min≈64px. These tuned peers have lower recall; their latency tradeoff stays visible."
    ),
    (
        "NAPA32 has a lower tail but misses more small tags. Frame-parallel throughput was not"
        " rerun and is not claimed here."
    ),
    "",
    "## Quality",
    "",
    f"Full replay source: `{qname}.csv`.",
    "",
    "| NAPA configuration | Found / 3486 | Wrong IDs | Duplicates |",
    "| --- | ---: | ---: | ---: |",
]
for b, v in counts.items():
    lines.append(f"| {b} | {v['found']} | {v['false_ids']} | {v['duplicates']} |")
lines += [
    "",
    ("For the unchanged peers, the previous same-corpus measurements found AprilTag d1 3,483/3,486, Nano 3,464,"),
    ("Locus standard 3,419 and Locus d2 3,273. These are reused measurements, not new full peer replays. All 1,080"),
    ("baseline corpus/stress truth, ID, corner-error and ordering rows match; all 540 range-image hashes match."),
    (
        "The corpus renderer differs from the earlier commit only in the expanded range-size "
        "list, for which the separate 18-size replay is used."
    ),
    "",
    ("Common-tag corner errors below reuse that verified reference scope; geometry is aligned before scoring."),
    "",
    "| Peer | Common tags | NAPA median / p95 / max px | Peer median / p95 / max px |",
    "| --- | ---: | ---: | ---: |",
]
for x in quality_pairs:
    if x["resolution"] == "all" and x["versus"] in (
        "apriltag3-d1",
        "apriltag3-d2",
        "aruco-nano",
        "locus",
        "opencv-subpix",
    ):
        a = x["baseline"]
        b = x["other"]
        lines.append(
            f"| {x['versus']} | {x['common_tags']} | {a['median_px']:.4f} / {a['p95_px']:.4f} / {a['max_px']:.4f} "
            f"| {b['median_px']:.4f} / {b['p95_px']:.4f} / {b['max_px']:.4f} |"
        )
lines += [
    "",
    (
        "NAPA retains the motion-blur worst-corner weakness against upstream AprilTag. Its "
        "median and p95 are better on common tags; that does not erase the worse maximum."
    ),
    "",
    "## Real images and limits",
    "",
    (
        f"The {real_ours['images']} AprilTag test JPEGs have reference regression output, not annotated truth. "
        f"NAPA16 takes {real_times}ms median per image and agrees with {real_ours['matched']}/"
        f"{real_ours['reference_tags']} reference detections ({real_ours['detections']} detections total). "
        f"AprilTag d1 agrees with {real['apriltag3-d1']['matched']}/{real['apriltag3-d1']['reference_tags']}, "
        f"Nano {real['aruco-nano']['matched']}/{real['aruco-nano']['reference_tags']}. "
        "No false-positive/recall claim is made for these images. No suitable annotated camera recording was available."
    ),
    "",
    ("The benchmark is finite, synthetic, and developed in this repository. A new seed does not make it independent"),
    (
        "of the renderer. No public performance table was changed. Measure on actual camera "
        "footage before relying on a specific latency or range."
    ),
    "",
    "## Adapter overhead",
    "",
    (
        f"For empty, one-tag and 27-tag 720p examples, additional Python input/result handling over a direct ctypes "
        f"call is approximately {overhead_us[0]:.1f}/{overhead_us[1]:.1f}/{overhead_us[2]:.1f} microseconds at p50. "
        "This is not subtracted from reported detector times. The direct ctypes call still has its own call overhead. "
        "RapidTag copies each input frame; that required copy remains timed. "
        "See adapter_overhead.json for the paired samples."
    ),
    "",
    "## Provenance and checks",
    "",
    (
        "- Final timed core hash is in `final_latency.json`; source/build hashes and "
        "intermediate snapshots are in `source_manifest.json`."
    ),
    (
        "- Earlier diagnostic timings preceded terminal-crop guards and metadata packing. "
        "Their original hashes remain intact. Final timing/resource runs use the final core; "
        "exact pre/post comparisons cover both seeds and all five strides."
    ),
    (
        f"- {test_summary}; Ruff passes. AddressSanitizer/UndefinedBehaviorSanitizer pass {sanitizer_calls} "
        "native calls over odd/padded shapes, sample strides and three families."
    ),
    (
        "- Exact ordered outputs matched on two seeds and all five strides after component "
        "packing. Reproduced small-crop, quiet-zone and tilted-tag regressions have focused "
        "tests."
    ),
    (
        "- Reproduction commands, pinned dependency setup and raw CSV/JSON files are linked "
        "from `commands.md`. No per-frame executable launches, detector initialization, GPU "
        "processing or corner-refinement disabling supply the reported wins."
    ),
    "",
]
(R / "report.md").write_text("\n".join(lines))
print("report generated using", qname)
