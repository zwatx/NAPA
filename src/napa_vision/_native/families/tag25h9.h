/* tag25h9 codewords in Napa's bit order (see gen_families.py): data cells row-major over the tag, most
 * significant bit first, a set bit for a white cell of the printed tag. Generated from upstream AprilTag
 * tag25h9.c at commit b7c0ebe9aa20f82ec7a828579004f9e706bfecd9; do not edit. Upstream notice:
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
static const int8_t napa_tag25h9_x[25] = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
static const int8_t napa_tag25h9_y[25] = {1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5};
static const uint64_t napa_tag25h9_codes[35] = {
    0x11fa755ULL, 0x0db164fULL, 0x02da1bdULL, 0x16726afULL, 0x0e650e9ULL, 0x1ba2558ULL,
    0x0cfd5dcULL, 0x1ab74c1ULL, 0x1ce8abbULL, 0x022caf6ULL, 0x1c93702ULL, 0x1049b14ULL,
    0x0ea27b6ULL, 0x130e14fULL, 0x0973035ULL, 0x082836dULL, 0x1580f8dULL, 0x103b372ULL,
    0x046ef9dULL, 0x120b8eaULL, 0x1147df4ULL, 0x07fd90bULL, 0x08d5276ULL, 0x144e803ULL,
    0x0685587ULL, 0x04be1c2ULL, 0x121c512ULL, 0x01e42cdULL, 0x14a9e26ULL, 0x1baf982ULL,
    0x1fd1d34ULL, 0x0141679ULL, 0x116a0beULL, 0x01b8ddbULL, 0x0e7133aULL,
};
#define NAPA_TAG25H9_FAMILY {"tag25h9", 7, 9, 0, 9, 25, 35, napa_tag25h9_x, napa_tag25h9_y, napa_tag25h9_codes}
