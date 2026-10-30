"""Regenerate full-corpus tables from saved measurements; no detector library is loaded."""

import argparse
import csv
import hashlib
import json
import re
import statistics as st
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from benchmarks.cpu_comparison import common_corners  # noqa: E402

LABELS = {
    "napa": "NAPA32 (default)",
    "napa-mtp16": "NAPA16 (tuned)",
    "wpilib-d2": "WPILib AprilTag d2",
    "apriltag3-d2": "AprilTag d2",
    "apriltag3-d1": "AprilTag d1",
    "locus": "Locus standard",
    "locus-d2": "Locus d2",
    "opencv": "OpenCV ArUco",
    "opencv-subpix": "OpenCV SUBPIX",
    "opencv-aruco3": "ArUco3 full",
    "opencv-aruco3-min32": "ArUco3 min≈64 px",
    "opencv-aruco3-half16": "ArUco3 min≈32 px",
    "aruco-nano": "ArUco Nano",
    "rapidtag": "RapidTag",
    "rapidtag-subpix": "RapidTag SUBPIX",
}
DOC_LABELS = {
    "napa": "NAPA",
    "napa-mtp16": "NAPA, `min_tag_px=16`",
    "wpilib-d2": "AprilTag, WPILib's defaults",
    "apriltag3-d2": "AprilTag, decimate 2",
    "apriltag3-d1": "AprilTag, full resolution",
    "locus": "Locus",
    "locus-d2": "Locus, decimate 2",
    "opencv": "OpenCV ArUco",
    "opencv-subpix": "OpenCV ArUco, subpixel",
    "opencv-aruco3": "ArUco3",
    "opencv-aruco3-min32": "ArUco3, tags over 64 px",
    "opencv-aruco3-half16": "ArUco3, tags over 32 px",
    "aruco-nano": "ArUco Nano",
    "rapidtag": "RapidTag",
    "rapidtag-subpix": "RapidTag, subpixel",
}
# The README table leaves out the two configurations that differ from a listed one only by skipping corner refinement.
README_ROWS = [b for b in DOC_LABELS if b not in ("napa-mtp16", "opencv", "rapidtag")]
OWN = ("napa", "napa-mtp16")
LINK = "benchmarks/results/2026-10-30_cpu_full_corpus"


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| --- |" + " ---: |" * (len(headers) - 1)]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def values(row, field):
    return [int(x) for x in row[field].split(";")]


def stats(rows):
    ns = [v for r in rows for v in values(r, "samples_ns")]
    cpu = [v for r in rows for v in values(r, "cpu_ns")]
    errors = [float(v) for r in rows for v in r["corner_errors_px"].split(";") if v]
    return {
        "frames": len(rows),
        "tags": sum(int(r["tags"]) for r in rows),
        "found": sum(int(r["true_positive"]) for r in rows),
        "false_ids": sum(int(r["false_id"]) for r in rows),
        "duplicates": sum(int(r["duplicate"]) for r in rows),
        "p50_p95_p99_ms": (np.percentile(ns, [50, 95, 99]) / 1e6).tolist(),
        "cpu_mean_ms": float(np.mean(cpu)) / 1e6,
        "corner_median_p95_max_px": np.percentile(errors, [50, 95, 100]).tolist(),
    }


def times(value):
    """Format a speed ratio the way the public docs print it."""
    if round(value, 2) == 1:
        return "1×"
    return f"{value:.0f}×" if value >= 10 else f"{value:.1f}×" if value >= 1 else f"{value:.2f}×"


def reflow(text):
    """Rewrap prose paragraphs to 120 columns, leaving headings, tables, and lists alone."""
    blocks = []
    for block in text.split("\n\n"):
        if block.strip() and not block.lstrip().startswith(("|", "#", "-", "```")):
            block = textwrap.fill(" ".join(block.split()), 120, break_long_words=False, break_on_hyphens=False)
        blocks.append(block)
    return "\n\n".join(blocks)


def per_frame(rows, numerator, denominator):
    """Per-frame ratios of two backends' median call times, keyed by frame."""
    medians = defaultdict(dict)
    for r in rows:
        medians[r["frame"]][r["backend"]] = st.median(values(r, "samples_ns"))
    return {frame: m[numerator] / m[denominator] for frame, m in medians.items()}


def guide_summary_table(summary):
    return table(
        ["Detector", "Median ms", "p95 ms", "p99 ms", "Tags found", "Wrong IDs"],
        [
            [
                DOC_LABELS[b],
                *[f"{v:.3f}" for v in s["p50_p95_p99_ms"]],
                f"{s['found']}/{s['tags']}",
                s["false_ids"],
            ]
            for b, s in summary.items()
        ],
    )


def summary_table(summary):
    return table(
        ["Detector / configuration", "p50 ms", "p95 ms", "p99 ms", "CPU mean ms", "Found / tags", "Wrong IDs", "Dup"],
        [
            [
                LABELS[b],
                *[f"{v:.3f}" for v in s["p50_p95_p99_ms"]],
                f"{s['cpu_mean_ms']:.3f}",
                f"{s['found']}/{s['tags']}",
                s["false_ids"],
                s["duplicates"],
            ]
            for b, s in summary.items()
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-docs", action="store_true", help="Also replace the public README comparison sections")
    args = parser.parse_args()
    measured = json.loads((R / "matrix.json").read_text())
    protocol = json.loads((R / "protocol.json").read_text())
    with (R / "matrix.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert not measured["skipped_backends"]
    assert set(measured["backends"]) == set(LABELS) == set(protocol["backends"])
    cfg = measured["run_config"]
    assert (cfg["frames"], cfg["stress_frames"], cfg["range_cases"], cfg["repetitions"], cfg["warmup"]) == (
        50,
        50,
        50,
        10,
        1,
    )
    assert cfg["only_frames"] is None and cfg["seed"] == cfg["order_seed"] == 2468
    assert measured["machine"]["affinity"] == [2]
    for phase in measured["threads_observed"].values():
        assert all(t["cpus"] == [2] for t in phase)
    for b in OWN:
        assert measured["backends"][b]["core_library_sha256"] == [protocol["napa_binary_sha256"]]
    grouped = defaultdict(dict)
    for row in rows:
        b, frame = row["backend"], row["frame"]
        assert frame not in grouped[b]
        grouped[b][frame] = row
        assert row["stable_across_repetitions"] == "True"
        for field in ("samples_ns", "cpu_ns"):
            assert len(values(row, field)) == 10 and min(values(row, field)) > 0
    expected = {
        f"{scene}-{res}-{i:03d}"
        for scene in cfg["scenarios"] + cfg["stress"]
        for res in protocol["resolutions"]
        for i in range(50)
    } | {f"range-640x480-{i:03d}" for i in range(900)}
    assert len(rows) == 67500 and len(expected) == 4500
    for b in LABELS:
        assert set(grouped[b]) == expected
        for name, row in grouped[b].items():
            assert all(row[k] == grouped["napa"][name][k] for k in ("image_sha256", "tags", "expected_ids"))
    max_truth = max(int(r["tags"]) for r in rows)
    max_own = max(len(r["detected_ids"].split(";")) if r["detected_ids"] else 0 for r in rows if r["backend"] in OWN)
    assert max_truth < 64 and max_own < 64
    scopes = {
        "main": [r for r in rows if r["scenario"] in cfg["scenarios"]],
        "stress": [r for r in rows if r["scenario"] in cfg["stress"]],
        "range": [r for r in rows if r["scenario"] == "range"],
    }
    summaries = {
        scope: {b: stats([r for r in rs if r["backend"] == b]) for b in LABELS} for scope, rs in scopes.items()
    }
    for scope, size in (("main", 3000), ("stress", 600), ("range", 900)):
        assert all(s["frames"] == size for s in summaries[scope].values())
    ratios = {}
    for scope, rs in scopes.items():
        ratios[scope] = {}
        for own in OWN:
            base = {r["frame"]: st.median(values(r, "samples_ns")) for r in rs if r["backend"] == own}
            ratios[scope][own] = {}
            for b in LABELS:
                vs = [st.median(values(r, "samples_ns")) / base[r["frame"]] for r in rs if r["backend"] == b]
                ratios[scope][own][b] = {
                    "median": st.median(vs),
                    "minimum": min(vs),
                    "slower_frames": sum(v < 1 for v in vs),
                }
    corners = {b: common_corners(scopes["main"], b) for b in OWN}
    analysis = {
        "scopes": summaries,
        "paired_ratios": ratios,
        "common_corners_main": corners,
        "max_truth_per_frame": max_truth,
        "max_napa_detections": max_own,
        "files_sha256": {
            n: hashlib.sha256((R / n).read_bytes()).hexdigest() for n in ("matrix.json", "matrix.csv", "protocol.json")
        },
    }
    (R / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    paired_rows = []
    for b in LABELS:
        if b in OWN:
            continue
        cells = [LABELS[b]]
        for own in OWN:
            v = ratios["main"][own][b]
            cells += [f"{v['median']:.2f}× ({v['minimum']:.5f}×)", v["slower_frames"]]
        paired_rows.append(cells)
    paired = table(
        [
            "Competitor",
            "Vs NAPA32: median (minimum)",
            "NAPA32 slower / 3000",
            "Vs NAPA16: median (minimum)",
            "NAPA16 slower / 3000",
        ],
        paired_rows,
    )
    main_table = summary_table(summaries["main"])
    quality_rows = [
        [LABELS[b], *[f"{v:.4f}" for v in s["corner_median_p95_max_px"]]] for b, s in summaries["main"].items()
    ]
    quality = table(["Detector", "Median worst-corner px", "p95 px", "Maximum px"], quality_rows)
    common_lines = []
    for own in OWN:
        common_lines += [
            f"### Common tags: {LABELS[own]}",
            "",
            table(
                ["Peer", "Common tags", "NAPA median / p95 / max px", "Peer median / p95 / max px"],
                [
                    [
                        LABELS[x["versus"]],
                        x["common_tags"],
                        " / ".join(f"{x['baseline'][k]:.4f}" for k in ("median_px", "p95_px", "max_px")),
                        " / ".join(f"{x['other'][k]:.4f}" for k in ("median_px", "p95_px", "max_px")),
                    ]
                    for x in corners[own]
                    if x["resolution"] == "all" and x["versus"] not in OWN
                ],
            ),
            "",
        ]
    pooled_win = all(
        all(
            x < y
            for x, y in zip(
                summaries["main"][own]["p50_p95_p99_ms"], summaries["main"][b]["p50_p95_p99_ms"], strict=True
            )
        )
        for own in OWN
        for b in LABELS
        if b not in OWN
    )
    findings = (
        "Both NAPA options have lower pooled main-corpus p50/p95/p99 than every competitor configuration. "
        if pooled_win
        else "Pooled main-corpus percentile results are mixed; compare each configuration in the table. "
    )
    default_losses = sum(ratios["main"]["napa"][b]["slower_frames"] for b in LABELS if b not in OWN)
    findings += (
        "NAPA32 has lower per-frame medians on every main-corpus image versus every competitor. "
        if default_losses == 0
        else f"NAPA32 has {default_losses} pairwise main-frame latency losses. "
    )
    loss_text = [
        f"{LABELS[b]}: {ratios['main']['napa-mtp16'][b]['slower_frames']}"
        for b in LABELS
        if b not in OWN and ratios["main"]["napa-mtp16"][b]["slower_frames"]
    ]
    findings += "NAPA16 slower-frame counts are " + ("; ".join(loss_text) or "zero for all peers") + ". "
    findings += "These are observed comparisons, not significance tests; near-ties remain included."
    method = (
        "One fixed shared corpus, recorded in protocol.json before timing, with 15 ordinary backend options: "
        "NAPA32/default (`napa`), NAPA16 (`napa-mtp16`) and 13 peer configurations. No frame filters or "
        "selection based on detector outputs. Main: 20 scenarios × 50 renders × three resolutions = 3,000 frames. "
        "Stress: four × 50 × three = 600 frames. Range: 18 sizes × 50 = 900 frames. Groups are reported separately. "
        "Each detector was constructed once and had one warm-up plus ten shuffled timed calls on every image; "
        "the first image also has an untimed lifecycle probe. Seed/order seed 2468. Same grayscale tag36h11 images, "
        "existing scoring, pixel-origin normalization and corner conventions. Required input copies and result "
        "conversion remain timed; construction, rendering, scoring and startup are excluded."
    )
    report = [
        "# Fixed full-corpus CPU comparison — 2026-10-30",
        "",
        method,
        "",
        "## Machine, build and effective settings",
        "",
        "Intel Core i7-12700H laptop, Ubuntu 26.04 x86_64. CPU2 is a verified P-core, with SMT sibling CPU3 monitored. "
        "Each backend is configured for one worker; every observed OS thread is confined to CPU2. "
        "Frozen native source commit: `f7a7adfe27ca701aee75561be924f64b83bdf442`; rebuilt core SHA256 "
        f"`{protocol['napa_binary_sha256']}`. Both NAPA options use the public adapter and retain refinement "
        "and correction settings. Maximum placed/detected NAPA tags per frame are both 35, "
        "below the 64-output capacity.",
        "",
        "AprilTag d1/d2 denotes decimation one/two. Locus d2 is its standard profile with decimation two. "
        "OpenCV/RapidTag defaults do not refine corners; SUBPIX variants are separate. ArUco3 full-scale "
        "does not refine at its default ratio despite the SUBPIX flag; its two half-scale configurations "
        "have effective minimum sizes of approximately 64 and 32 px. Exact versions, builds and effective "
        "settings are recorded in matrix.json. Peer binary hashes are not a complete inventory.",
        "",
        "## Main corpus: 3,000 frames",
        "",
        "Each backend has 30,000 timed calls and the same 5,521 expected tags. CPU mean is process CPU time "
        "per detector call. Detection counts are scored once per image, not multiplied by repetitions.",
        "",
        main_table,
        "",
        findings,
        "",
        paired,
        "",
        "NAPA32 found 5,513/5,521 tags; NAPA16 found 5,519/5,521. Both reported one wrong ID on "
        "`decoys-640x480-042`. Nano found 5,517 with no wrong IDs. NAPA16 missed two tags on "
        "`low-contrast-1280x720-046` and `-047`; the two NAPA options do not have strictly nested detection sets.",
        "",
        "## Corner quality on the main corpus",
        "",
        "Worst-corner distance per matched tag after geometric alignment. These medians condition on tags "
        "found; compare common-tag tables for a matched quality comparison. Native canonical corner order "
        "is recorded separately in matrix.json/CSV and is not the geometric error metric.",
        "",
        quality,
        "",
        *common_lines,
        "## Stress: 600 frames",
        "",
        summary_table(summaries["stress"]),
        "",
        "## Range: 900 frames",
        "",
        summary_table(summaries["range"]),
        "",
        "## Measurement conditions and limits",
        "",
        "The fixed run lasted approximately 2h 56m. The 2,086 monitor samples show CPU2 frequency "
        "2.799–4.578 GHz (median 3.952 GHz); package temperature median 91°C, maximum 97°C. CPU3 busy "
        "fraction was median 0%, p95 3%, maximum 38%. Background applications were retained. "
        "Quarter-wise median frequencies stayed approximately 3.92–3.99 GHz; the observations do not "
        "by themselves establish throttling. All outliers and all preselected frames remain included.",
        "",
        "This interleaved single-core run measures call latency and process CPU, not backend-isolated RSS "
        "or multicore performance. Earlier "
        "[isolated resource measurements](../2026-10-29_cpu_optimization/report.md#cpu-and-working-memory) "
        "use separate 96-frame sets and retain their original provenance; they are not full-corpus resource results. "
        "The earlier 149-frame set is a development diagnostic only and is superseded for headline latency/quality. "
        "This synthetic renderer and one host do not establish real-camera accuracy or a universal winner.",
        "",
        "## Integrity and reproduction",
        "",
        "Independent read-only audit approved methodology and completed data: 67,500 rows; 4,500 shared "
        "images per backend; complete scenario/index coverage; equal hashes, truth and frame properties; "
        "ten positive wall/CPU samples each; no skips or unstable rows; frozen native/benchmark hashes; "
        "matching preflight settings. The monitor is not a continuous process inventory, so absolute "
        "absence of other jobs throughout cannot be proved from it.",
        "",
        "[Protocol](protocol.json), [commands](commands.md), [raw JSON](matrix.json), [raw CSV](matrix.csv), "
        "[thermal/load monitor](monitor.tsv), [adapter validation](validate.json), [analysis](analysis.json). "
        "Run `work/venv/bin/python benchmarks/results/2026-10-30_cpu_full_corpus/make_report.py` to regenerate "
        "this report without detecting or loading a native library. Add `--write-docs` to refresh README tables.",
        "",
    ]
    (R / "report.md").write_text("\n".join(report))
    main = summaries["main"]
    range_counts = {
        b: {
            size: sum(
                int(r["true_positive"])
                for r in scopes["range"]
                if r["backend"] == b and float(r["detail_side_px"]) == size
            )
            for size in protocol["range"]["sizes_px"]
        }
        for b in LABELS
    }

    def smallest_always_found(b):
        sizes = sorted(protocol["range"]["sizes_px"], reverse=True)
        found = None
        for size in sizes:
            if range_counts[b][size] != 50:
                break
            found = size
        return found

    reach = {b: smallest_always_found(b) for b in ("napa", "napa-mtp16", "wpilib-d2", "apriltag3-d1")}
    assert reach["napa-mtp16"] == min(protocol["range"]["sizes_px"])
    speed_rows = []
    for b in README_ROWS:
        m, label = main[b], DOC_LABELS[b]
        speed = st.median(per_frame(scopes["main"], "wpilib-d2", b).values())
        cells = [
            times(speed),
            f"{100 * m['found'] / m['tags']:.1f}%",
            f"{m['corner_median_p95_max_px'][0]:.3f} px",
            m["false_ids"],
        ]
        if b == "napa":
            label, cells = f"**{label}**", [f"**{c}**" if i < 3 else c for i, c in enumerate(cells)]
        speed_rows.append([label, *cells])
    speed_table = table(
        [
            "Detector",
            "Speed vs. WPILib's defaults (median frame)",
            "Tags found",
            "Worst-corner error (median)",
            "Wrong IDs",
        ],
        speed_rows,
    )
    assert default_losses == 0
    closest = min((b for b in LABELS if b not in OWN), key=lambda b: ratios["main"]["napa"][b]["median"])
    resolution_rows = []
    for res in protocol["resolutions"]:
        rs = [r for r in scopes["main"] if r["resolution"] == res]
        cells = [res.replace("x", "×")]
        for b in ("wpilib-d2", "apriltag3-d1"):
            v = list(per_frame(rs, b, "napa").values())
            cells.append(f"{times(st.median(v))} ({times(min(v))})")
        resolution_rows.append(cells)
    resolution_table = table(
        ["Resolution", "vs. WPILib's defaults, median (slowest frame)", "vs. full resolution, median (slowest frame)"],
        resolution_rows,
    )
    tuned = ratios["main"]["napa-mtp16"]
    tuned_losers = [b for b in LABELS if b not in OWN and tuned[b]["slower_frames"]]
    assert set(tuned_losers) == {"locus-d2", "opencv-aruco3-min32", "opencv-aruco3-half16", "aruco-nano"}
    tuned_loss_frames = {
        frame for b in tuned_losers for frame, ratio in per_frame(scopes["main"], b, "napa-mtp16").items() if ratio < 1
    }
    tuned_cost = st.median(per_frame(scopes["main"], "napa-mtp16", "napa").values())
    corner = {b: main[b]["corner_median_p95_max_px"][0] for b in ("napa", "apriltag3-d1", "wpilib-d2")}
    assert corner["napa"] < corner["apriltag3-d1"] / 2 and 4.5 < corner["wpilib-d2"] / corner["napa"] < 5.5
    assert main["napa"]["false_ids"] == 1
    d1_common = next(x for x in corners["napa"] if x["versus"] == "apriltag3-d1" and x["resolution"] == "all")
    stress = summaries["stress"]
    root = f"""## Compared with other detectors

The benchmark is 3,000 rendered frames holding 5,521 tags at 640×480, 1280×720, and 1920×1080, each with exact ground
truth. Every detector ran single-threaded on the same CPU core, on the same frames.

{speed_table}

NAPA was faster than every other detector on every one of the 3,000 frames. The closest was
{DOC_LABELS[closest]}, and NAPA's smallest lead over it on any frame was
{times(ratios["main"]["napa"][closest]["minimum"])}.
NAPA's one wrong ID was a tag-shaped decoy with a random pattern. OpenCV ArUco and RapidTag without subpixel
refinement run at the same speed as above, with corners about a pixel off.

NAPA's lead over AprilTag grows with resolution:

{resolution_table}

**Range.** A tag looks smaller the farther away it is, so the smallest tag a detector finds sets how far it can see. In
the [range sweep](benchmarks/README.md#range), NAPA found every tag from {reach["napa"]} px up, AprilTag at WPILib's
defaults from {reach["wpilib-d2"]} px, and AprilTag at full resolution from {reach["apriltag3-d1"]} px. On the same
camera, NAPA sees about {reach["wpilib-d2"] / reach["napa"]:.1f}× as far as WPILib's defaults.

**Smaller tags.** `min_tag_px=16` found every tag in the range sweep, down to {reach["napa-mtp16"]} px, and
{main["napa-mtp16"]["found"]:,} of the 5,521 tags above. It takes about {tuned_cost:.0f}× as long as the default, which
still makes it {times(st.median(per_frame(scopes["main"], "wpilib-d2", "napa-mtp16").values()))} faster than WPILib's
defaults. ArUco Nano, ArUco3, and Locus at decimate 2 were faster than it on {len(tuned_loss_frames)} of the 3,000
frames.

**Corners.** NAPA's median worst-corner error is less than half of full-resolution AprilTag's and about a fifth of
WPILib's defaults, and on 95% of tags it was within {main["napa"]["corner_median_p95_max_px"][1]:.2f} px. Heavy motion
blur is the exception: NAPA's largest error was {d1_common["baseline"]["max_px"]:.1f} px, against
{d1_common["other"]["max_px"]:.1f} px for full-resolution AprilTag on the same tags.

**Stress.** A separate set of 600 harder frames has tags under 32 px, very heavy noise, and up to 80 tags per frame.
NAPA found {100 * stress["napa"]["found"] / stress["napa"]["tags"]:.0f}% of those tags with its default settings, and
all of them with `min_tag_px=16`.

On tag16h5, the 2023 FRC family, AprilTag at its default two-bit correction reported 581 wrong IDs in the
[family benchmark](benchmarks/README.md#tag-families), and NAPA reported none.

The frames are synthetic, and the timings come from one laptop CPU (Intel i7-12700H). To check NAPA on your own
camera, [score it on your own footage](docs/measuring.md). The [full report]({LINK}/report.md) has every
configuration's settings, latency percentiles, and raw data, and the [benchmark guide](benchmarks/README.md) has
results for every scene.

"""
    per_scene = []
    for res in protocol["resolutions"]:
        for mode in ("time", "quality"):
            output = []
            for scene in cfg["scenarios"]:
                rs = [r for r in scopes["main"] if r["resolution"] == res and r["scenario"] == scene]
                own = [r for r in rs if r["backend"] == "napa"]
                tags = sum(int(r["tags"]) for r in own)
                cells = []
                for b in LABELS:
                    members = [r for r in rs if r["backend"] == b]
                    if mode == "time":
                        cells.append(f"{st.median(v for r in members for v in values(r, 'samples_ns')) / 1e6:.3f} ms")
                    else:
                        n = sum(int(r["true_positive"]) for r in members)
                        wrong = sum(int(r["false_id"]) for r in members)
                        cells.append(str(n) + (f" ({wrong})" if wrong else ""))
                output.append([scene, 50, tags, *cells] if mode == "time" else [scene, tags, *cells])
            title = "median detector call" if mode == "time" else "tags found (wrong IDs in brackets)"
            headers = ["Scene", "Frames", "Tags"] if mode == "time" else ["Scene", "Tags"]
            per_scene += [f"### {res}: {title}", "", table([*headers, *DOC_LABELS.values()], output), ""]
    range_rows = []
    for b in LABELS:
        cells = []
        for size in protocol["range"]["sizes_px"]:
            rs = [r for r in scopes["range"] if r["backend"] == b and float(r["detail_side_px"]) == size]
            assert len(rs) == 50
            cells.append(sum(int(r["true_positive"]) for r in rs))
        range_rows.append([DOC_LABELS[b], *cells])
    assert all(s["duplicates"] == 0 for s in summaries["main"].values())
    guide_latest = (
        "## Latest results\n\nThese tables come from the "
        "[2026-10-30 run](results/2026-10-30_cpu_full_corpus/report.md), which gave every detector the same "
        "3,000 frames: 50 per scene at each resolution. Each detector had one warm-up and ten timed calls per frame, "
        "in shuffled order, on one performance core of an Intel i7-12700H. Each time is the median of a scene's "
        "timed calls.\n\n"
        "AprilTag at WPILib's defaults is `robotpy-apriltag` as WPILib configures it: decimate 2, a 300 px minimum "
        "cluster, and a 45° critical angle. The other AprilTag rows use the upstream C library's defaults. OpenCV "
        "ArUco and RapidTag refine corners only in their subpixel rows. The two limited ArUco3 rows detect at half "
        "scale, which sets the smallest tag they can find. The [report](results/2026-10-30_cpu_full_corpus/report.md) "
        "lists every configuration's exact settings, and compares NAPA's corners with each detector's on the tags both "
        "found.\n\n"
        + "\n".join(per_scene)
        + "\nNo detector reported the same tag twice. The same run's 600 stress frames (tags under 32 px, noise "
        "σ 15–30, 30–80 tags per frame, and noisy clutter) are scored separately:\n\n"
        + guide_summary_table(summaries["stress"])
        + "\n\n"
    )
    guide_range = (
        "## Range\n\nTags found out of 50 at each black-square size, from the same run: 900 frames at 640×480, "
        "with light noise and up to 15° of tilt.\n\n"
        + table(["Detector", *map(str, protocol["range"]["sizes_px"])], range_rows)
        + f"\n\nWith its default `min_tag_px=32`, NAPA found every tag from {reach['napa']} px up. "
        "With `min_tag_px=16`, it found every tag at every size tested. Real cameras add blur, noise, and "
        "steeper angles, so measure range on your own.\n\n"
    )
    if args.write_docs:
        for path in (ROOT / "README.md", ROOT / "docs/pypi.md"):
            s = path.read_text()
            start, end = s.index("## Compared with other detectors"), s.index("## How it works")
            block = root
            if path.name == "pypi.md":
                block = re.sub(
                    r"\]\((?!https?://)([^)]+)\)",
                    lambda m: "](https://github.com/zwatx/NAPA/blob/master/" + m[1] + ")",
                    block,
                )
            path.write_text(s[:start] + reflow(block) + s[end:])
        path = ROOT / "benchmarks/README.md"
        s = path.read_text()
        start, end = s.index("## Latest results"), s.index("## Tag families")
        s = s[:start] + reflow(guide_latest) + reflow(guide_range) + s[end:]
        start = s.index("\n\n", s.index("and pose solving")) + 2
        end = s.index("## What's in the corpus")
        s = (
            s[:start]
            + (
                "The latest results are from the [2026-10-30 run](results/2026-10-30_cpu_full_corpus/report.md), "
                "which ran NAPA and 13 configurations of other detectors on the same frames.\n\n"
            )
            + s[end:]
        )
        s = s.replace(
            "[Range table below](#range) uses a separate 30-tags-per-size replay.",
            "[Range table below](#range) uses that same 50-tags-per-size sweep from the fixed full run.",
        )
        s = s.replace(
            "Per-scene brackets contain wrong IDs; there were\n  no duplicates in the diagnostic comparison.",
            "Per-scene brackets contain wrong IDs; there were\n  no duplicates in the full main-corpus comparison.",
        )
        s = s.replace(
            "results/2026-10-29_cpu_optimization/report.md#quality",
            "results/2026-10-30_cpu_full_corpus/report.md#corner-quality-on-the-main-corpus",
        )
        path.write_text(s)
    print("Full-corpus report regenerated; public docs written:", args.write_docs)


if __name__ == "__main__":
    main()
