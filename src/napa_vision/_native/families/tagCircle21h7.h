/* tagCircle21h7 codewords in Napa's bit order (see gen_families.py): data cells row-major over the tag, most
 * significant bit first, a set bit for a white cell of the printed tag. Generated from upstream AprilTag
 * tagCircle21h7.c at commit b7c0ebe9aa20f82ec7a828579004f9e706bfecd9; do not edit. Upstream notice:
 *
 * Copyright (C) 2013-2016, The Regents of The University of Michigan.
 * All rights reserved.
 * This software was developed in the APRIL Robotics Lab under the
 * direction of Edwin Olson, ebolson@umich.edu. This software may be
 * available under alternative licensing terms; contact the address above.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Regents of The University of Michigan.
 */
static const int8_t napa_tagCircle21h7_x[21] = {1, 2, 3, -2, 1, 2, 3, 6, -2, 1, 2, 3, 6, -2, 1, 2, 3, 6, 1, 2, 3};
static const int8_t napa_tagCircle21h7_y[21] = {-2, -2, -2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 6, 6, 6};
static const uint64_t napa_tagCircle21h7_codes[38] = {
    0x02be65ULL, 0x1a384cULL, 0x0de4b9ULL, 0x1461c8ULL, 0x1b06aeULL, 0x128987ULL,
    0x0a45f2ULL, 0x19fa9dULL, 0x1137e4ULL, 0x1e95f7ULL, 0x165bd6ULL, 0x0e6b3cULL,
    0x060655ULL, 0x1adb01ULL, 0x1925ebULL, 0x086106ULL, 0x138314ULL, 0x074cb7ULL,
    0x117430ULL, 0x0bb73cULL, 0x063eabULL, 0x1ec544ULL, 0x13e1a0ULL, 0x0b99c9ULL,
    0x0c8b40ULL, 0x1757bdULL, 0x1eb916ULL, 0x10b3deULL, 0x042dc6ULL, 0x15bb20ULL,
    0x19a512ULL, 0x009456ULL, 0x121e91ULL, 0x0ad25dULL, 0x0774d2ULL, 0x03688aULL,
    0x0c3d68ULL, 0x05b697ULL,
};
#define NAPA_TAGCIRCLE21H7_FAMILY {"tagCircle21h7", 5, 9, 1, 7, 21, 38, napa_tagCircle21h7_x, napa_tagCircle21h7_y, napa_tagCircle21h7_codes}
