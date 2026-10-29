# How NAPA works

Upstream AprilTag processes the whole frame before it looks for tags. It thresholds every pixel, segments the image
into regions, and fits a quadrilateral to every region's outline. NAPA finds likely tags on a sparse grid of pixels
first, skips anything without contrast, and only does full-resolution work on the tags themselves.

![NAPA on one frame, stage by stage: it samples every third pixel, skips flat tiles, turns dark samples into runs and blobs, drops shapes that can't be tags, and fits edges and decodes only the tags, at full resolution.](napa-pipeline.webp)

*One synthetic frame, stage by stage. Every overlay is computed from the frame by a port of NAPA's pipeline.*

## 1. Sample a sparse grid

A tag36h11 tag is 8 cells across, and its black border is one cell wide. NAPA reads only the pixels on a grid spaced ¾
of the smallest cell apart, which always puts a sample in the dark core of the border, even when blur greys the cell's
edges. With the default 32 px minimum tag size, that's every third pixel in each direction, or one pixel in nine.

Larger tags are found on coarser grids (2×, 4×, 8×, and up). Each is built from the one before rather than from the
image, and together they add about a third to the cost. Each tag is handled on the grid where it's between about 16
and 64 samples across. On a finer grid, the middle of its border is too far from any white to stand out. On a coarser
one, its one-cell white margin can fall between samples, and the tag merges with a dark background.

## 2. Skip regions without contrast

The grid is divided into small tiles, and NAPA records each tile's darkest and brightest sample. A tile without
contrast can't contain a tag's edge, so NAPA skips it without thresholding or segmenting it. The contrast required
rises with sensor noise, which NAPA measures from the frame itself, region by region, so a noisy or textured patch
doesn't hide tags elsewhere in the frame. On a very noisy frame, each grid sample averages the block of pixels it
stands for, which reduces the noise by the grid spacing.

## 3. Group dark samples into blobs

Every pass over the grid works on whole rows with vector instructions. One SSE2 or NEON instruction compares 16
samples with their thresholds and packs the results into a bitmask. Runs of dark samples come out of the masks with
one count-trailing-zeros instruction per run, and union-find joins runs that touch into blobs. The cost follows the
number of runs, not the number of dark pixels. Gathering every third pixel uses a byte shuffle (SSSE3, chosen at run
time) or NEON's de-interleaving load.

## 4. Reject non-tags before measuring them

Each blob of plausible size gets a rough quadrilateral from its extreme points. A blob darker overall than any real
code allows is dropped: every tag36h11 code has at least 10 white cells. Then NAPA:

1. Fits each side to its edge at full resolution.
2. Decodes straight away with a loose tolerance, reading the border first. Text, texture, and checkerboards are
   rejected here, before any expensive refinement.
3. Refines each side coarse to fine, then measures the edge every 1.5 px along it. Each measurement is the centre of
   the edge's gradient, which is exact for real edges, where each pixel averages the light falling on it. Where blur
   is wider than about half a cell, NAPA instead locates the edge where the image crosses halfway between the black
   and white on either side.
4. Intersects the four lines to get the corners, and decodes the payload.

Each side's line fit drops samples that disagree with the rest of the edge, so a short obstruction along a side
doesn't pull the corners off.

Three kinds of image fool a plain edge fit, so NAPA checks for each:

- **A grey quiet zone on a bright background** makes the margin look dark, so the first outline is the margin's. If
  the decode fails, NAPA tries the outline's inner 8/10, where the black square would be.
- **A side fitted one cell out**, on the margin's outer edge, shows up as border cells that read white. NAPA moves
  that side in by a cell and refits it.
- **Clipped exposure** shrinks every black region by the same amount. The black border must be exactly one cell wide,
  so NAPA measures the shrinkage on the tag itself and undoes it.

## 5. Look up the ID

A tag36h11 payload has 36 bits, which must be checked against 587 codes in 4 rotations, allowing up to 3 wrong bits.
Any code within 3 bits of the payload must match at least one of its four 9-bit blocks exactly. NAPA looks those
blocks up in a table and compares only the few codes that share one, instead of all 2,348.

## Obstructions

NAPA corrects up to 3 wrong payload bits, where upstream AprilTag defaults to 2, so a hand, cable, or glare spot over
part of a tag often still decodes. It accepts a third corrected bit only when the three wrong cells touch, as they do
under a real obstruction. A random pattern that happens to fall within three bits of a code has its errors scattered,
so this keeps chance matches out. In the benchmark's obstruction sweep, 864 tags with 0–40% of the black square
covered by a white or grey patch, NAPA found 357 and AprilTag at decimate 2 found 330.

Every AprilTag detector finds the black border before it reads any bits, so none can find a tag whose border is hidden
across a whole side or around a corner.

## Limits

- **Tag size.** Tags smaller than `min_tag_px` may be missed. A steeply tilted tag counts by its narrowest side.
- **Quiet zone.** Tags need the white margin that printed tags have. A tag whose margin runs straight into dark, busy
  texture can merge with it and be missed. Steep tilt makes this likelier, because it narrows the margin.
- **Contrast.** Tags with less than `min_contrast` grey levels between their black and white are skipped.
- **Motion blur.** Blur longer than one cell (an eighth of the tag) is the hardest case for corner accuracy. Most
  corners stay within a tenth of a pixel, but in the benchmark's worst case one corner was 2.5 px off, where
  AprilTag's worst in that scene was 0.98 px.

[Tuning](detection.md#tuning) covers `min_tag_px` and `min_contrast`.
