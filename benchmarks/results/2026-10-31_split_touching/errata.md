# Random-payload filter correction

The saved `grid_decoys.py` encoded real tag cells as white bits and decoys as dark bits. Its distance filter therefore
compared decoys with the complements of real codes. The claim that every decoy is at least five bits from every
tag36h11 code is incorrect.

Replaying the original seed and draw order reproduces all 25,835 decoys. Of these, 62 are within four bits of a real
code, including 11 within three bits; none is within two. The recorded detection counts remain the results of that
run. They cannot establish whether its one false ID with three-bit piece correction came from a near-code decoy.

The original script, measurements and hashes are retained. For a corrected experiment, use `1 - tag_cells(...)` when
building the real-code payloads and set `split_touching=False` explicitly for the unsplit configuration. Current
NAPA defaults to automatic splitting, so an omitted setting no longer reproduces the original unsplit run.
