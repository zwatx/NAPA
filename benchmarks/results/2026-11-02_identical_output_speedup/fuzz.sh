#!/bin/bash
# fuzz.sh BASELINE_SOURCE CORPUS [EVERY] [VARIANTS]: link the baseline core (symbols renamed base_*) and this
# checkout's core into one AddressSanitizer/UBSan binary, and compare every detection field on random crops of
# every EVERY-th corpus frame (dump_corpus.py output) with odd sizes and padded strides, for three families and
# seven min_tag_px values. Each crop is allocated to its exact size, so reads past it fault.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
base=$1 corpus=$2 every=${3:-13} variants=${4:-4}
out=$(mktemp -d)
api="napa_create napa_create_family napa_destroy napa_detect napa_family_count napa_family_name napa_family_info
     napa_family_code napa_tag36h11_code PyInit__napa_core"
rename=""
for f in $api; do rename="$rename -D$f=base_$f"; done
flags="-O1 -g -std=c99 -fno-wrapv -I$root/src/napa_vision/_native -fsanitize=address,undefined -fno-sanitize-recover=undefined"
cc $flags $rename -c -o "$out/base.o" "$base"
cc $flags -c -o "$out/cand.o" "$root/src/napa_vision/_native/napa_core.c"
cc $flags -o "$out/fuzz" "$here/fuzz.c" "$out/base.o" "$out/cand.o" -lm -l:libzstd.so.1
"$out/fuzz" "$corpus" "$every" "$variants"
