# How NAPA works

Upstream AprilTag processes the whole frame first: threshold every pixel, segment, fit quads everywhere. NAPA works
the other way round. It finds likely tags on a sparse grid, then spends full-resolution work only on them.

![NAPA on one frame, stage by stage: it samples a sparse grid, skips flat tiles, groups dark samples into blobs, drops shapes that cannot be tags, then fits edges and decodes only the tags, at full resolution.](napa-pipeline.webp)

*One synthetic frame, stage by stage, at `min_tag_px=32`.*

## 1. Sample a sparse grid

A tag36h11 tag is 8 cells across, border one cell wide. Sampling every ¾ cell guarantees a sample lands in the
border's dark core, even with blur. At the default `min_tag_px=22` that is every other pixel (1 in 4). At 32, every
third (1 in 9).

Bigger tags are caught on coarser grids (2x, 4x, 8x…), each built from the last. Bands overlap, so a tilted or blurred
tag can surface on several grids. Your pixels are never modified.

## 2. Skip flat regions

The grid is tiled. Tiles without enough dark-to-bright range cannot contain a tag edge, so they are skipped whole.
The bar rises with measured sensor noise, region by region, so one noisy patch does not blind the rest of the frame.
On noisy frames samples average their neighbours, except on the finest grid, where that would erase tiny borders.

## 3. Group dark samples into blobs

Rows are processed with SIMD: one instruction compares 16 samples against their thresholds. Runs of dark samples are
extracted from bitmasks and joined by union-find. Cost follows the number of runs, not dark pixels.

Diagonal-only contacts get checked before joining, which separates most calibration-board tags that touch at a corner
for free. The harder contacts are what [`split_touching`](detection.md#calibration-boards) handles.

## 4. Reject non-tags early, then measure

Each blob gets a rough quad from its extreme points. Impossible shapes die fast: wrong size, wrong aspect, too dark
overall (every tag36h11 code has at least 10 white cells).

Survivors are decoded once with a loose fit. Text, texture, and checkerboards die here, cheaply. Only then does NAPA
refine: each side is measured every 1.5 px at full resolution, fitted as a line (outliers dropped, so short
obstructions do not drag corners), and corners come from intersecting the four lines. Sharp edges are located by their
gradient centre, blurred ones by their midpoint crossing.

A few shapes need repairs. Each is kept only if the tag still decodes to the same ID:

- **Bowed sides** (wide lenses): corners re-fit from short edge runs, or sides fit as curves.
- **Sides cut by the frame**: borrow the bow from the fully visible sides. Corners outside the frame are never re-fit.
- **Grey quiet zone**: if the outline was really the margin, retry at its inner eight tenths.
- **Side one cell out**: move it in a cell and re-fit.
- **Clipped exposure**: the border must be exactly one cell wide; measure the shrink on the tag and undo it.
- **Blur and tone curves**: blur on gamma-encoded cameras reads tags a few percent small. NAPA measures blur and tone
  per tag after decoding and shifts sides back. [`camera_gamma`](detection.md#camera-gamma) goes further with a value
  you supply.

Duplicates across grids are removed by centre distance, then by outline containment. Two real copies of the same ID
both survive.

## 5. Look up the ID

36 payload bits against 587 codes in 4 rotations, allowing 3 errors. Any code within 3 bits shares one of the
payload's four 9-bit blocks exactly, so a table lookup narrows 2,348 candidates to a handful.

## Obstructions

NAPA corrects up to 3 wrong bits, but a third bit counts only when the wrong cells touch, as under a finger or glare
spot. Random patterns have scattered errors, so chance matches stay out. In the obstruction sweep (864 tags, up to 40%
covered) NAPA found 357, WPILib's AprilTag 330.

No AprilTag detector can find a tag whose border is hidden across a whole side or around a corner. The border is found
before any bits are read.

## Limits

- Tags under `min_tag_px` on their narrowest side may be missed.
- Tags need their white quiet zone. Merged into dark texture, they disappear. Steep tilt narrows the margin and makes
  this likelier.
- Tags under `min_contrast` grey levels of contrast are skipped.
- Motion blur longer than one cell (⅛ of the tag) is the worst case for corners: 3.4 px worst in the published
  comparison. A successful decode is not proof of good geometry. Check on your camera.
