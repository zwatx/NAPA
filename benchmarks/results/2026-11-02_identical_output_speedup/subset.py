"""dump_corpus.py output -> raw subset for dump.c (the aarch64 and NAPA_NO_SIMD check):
frame 0 of each scene and resolution, case 0 of each range size.
Record: u16 name_len, name, u32 w, u32 h, w*h bytes."""

import struct
import sys

import zstandard

d = zstandard.ZstdDecompressor()
n = 0
with open(sys.argv[1], "rb") as fi, open(sys.argv[2], "wb") as fo:
    while hdr := fi.read(2):
        (nl,) = struct.unpack("<H", hdr)
        name = fi.read(nl)
        w, h, nt = struct.unpack("<III", fi.read(12))
        fi.read(4 * nt)
        (zs,) = struct.unpack("<I", fi.read(4))
        z = fi.read(zs)
        idx = int(name.rsplit(b"-", 1)[1])
        if (name.startswith(b"range") and idx % 50 == 0) or (not name.startswith(b"range") and idx == 0):
            fo.write(hdr + name + struct.pack("<II", w, h) + d.decompress(z))
            n += 1
print("frames", n)
