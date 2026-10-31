"""Replay the recorded baseline/fallback timing experiment from the repository root.

Inputs: work/external/frc/selected.json and its available PNGs; compiled baseline.so
and fallback-strict.so. This is reference agreement, not independently labelled accuracy.
Detailed spatial agreement is recorded separately in frc-agreement.json.gz.
"""

import gzip
import hashlib
import json
import os
import pathlib
import random
import sys
import time

import cv2
import numpy as np

sys.path[:0] = [".", "src"]
from benchmarks import cpu_comparison as cc  # noqa: E402
from benchmarks import performance_matrix as pm  # noqa: E402
from napa_vision.napa_detector import NapaDetector, _library  # noqa: E402

root = pathlib.Path("work/external")
artifact = pathlib.Path(__file__).parent
expected_missing = set(json.loads(gzip.decompress((artifact / "frc-agreement.json.gz").read_bytes()))["missing"])
selected = root / "frc/selected.json"
if selected.read_bytes() != (artifact / "frc-selection.json").read_bytes():
    raise ValueError("selection differs from the recorded protocol")
dets = {}
for label, lib in [("baseline", "baseline.so"), ("candidate", "fallback-strict.so")]:
    os.environ["NAPA_CORE_LIBRARY"] = str((root / lib).resolve())
    _library.cache_clear()
    for m in (16, 32):
        dets[(label, m)] = NapaDetector(min_tag_px=m, max_detections=1024)
rows = []
rng = random.Random(42)

for index, record in enumerate(json.loads((root / "frc/selected.json").read_text())):
    path = root / "frc" / pathlib.PurePosixPath(record["file_name"]).name
    if record["file_name"] in expected_missing:
        continue
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
        raise ValueError(f"missing or modified selected image: {path.name}")
    im = cv2.imread(str(path), 0)
    truth = [pm.TagTruth(d["tag_id"], np.array(d["corners"]) - 0.5) for d in record["detections"]]
    for det in dets.values():
        det.detect(im)
    samples = {k: [] for k in dets}
    found = {}
    for _rep in range(10):
        order = list(dets)
        rng.shuffle(order)
        for k in order:
            t = time.perf_counter_ns()
            obs = dets[k].detect(im)
            samples[k].append((time.perf_counter_ns() - t) / 1e6)
            found[k] = [(o.tag_id, o.corners) for o in obs]
    for k, ds in found.items():
        rows.append(dict(frame=index, backend=k[0], min_tag_px=k[1], times_ms=samples[k], **cc.score_detail(truth, ds)))
(root / "frc-paired-reproduction.json").write_text(json.dumps(rows))
for m in (16, 32):
    a = [r for r in rows if r["min_tag_px"] == m and r["backend"] == "baseline"]
    b = [r for r in rows if r["min_tag_px"] == m and r["backend"] == "candidate"]
    print(
        m,
        "TP",
        sum(r["true_positive"] for r in a),
        sum(r["true_positive"] for r in b),
        "FP",
        sum(r["false_id"] for r in a),
        sum(r["false_id"] for r in b),
        "p50_ratio",
        np.median([np.median(y["times_ms"]) / np.median(x["times_ms"]) for x, y in zip(a, b, strict=True)]),
        "exact",
        sum(x["matched"] == y["matched"] for x, y in zip(a, b, strict=True)),
        len(a),
        flush=True,
    )
