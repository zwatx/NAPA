/* tag16h5 codewords in Napa's bit order (see gen_families.py): data cells row-major over the tag, most
 * significant bit first, a set bit for a white cell of the printed tag. Generated from upstream AprilTag
 * tag16h5.c at commit b7c0ebe9aa20f82ec7a828579004f9e706bfecd9; do not edit. Upstream notice:
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
static const int8_t napa_tag16h5_x[16] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
static const int8_t napa_tag16h5_y[16] = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
static const uint64_t napa_tag16h5_codes[30] = {
    0xd8c4ULL, 0xa574ULL, 0x562cULL, 0x9da2ULL, 0x659eULL, 0xd6feULL,
    0x1acdULL, 0xa2e7ULL, 0x9a7fULL, 0xb6a8ULL, 0xd01cULL, 0xd50fULL,
    0x21b0ULL, 0x6ce2ULL, 0x4e31ULL, 0x08f5ULL, 0x3c90ULL, 0x2dc9ULL,
    0xc0a5ULL, 0xf162ULL, 0xec87ULL, 0xa9eaULL, 0x42fbULL, 0xb838ULL,
    0x3b97ULL, 0xb5ceULL, 0xfab5ULL, 0x0cabULL, 0x53e0ULL, 0x74f5ULL,
};
#define NAPA_TAG16H5_FAMILY {"tag16h5", 6, 8, 0, 5, 16, 30, napa_tag16h5_x, napa_tag16h5_y, napa_tag16h5_codes}
