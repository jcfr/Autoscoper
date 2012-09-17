// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// (1) Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of Brown University nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY “AS IS” WITH NO
// WARRANTIES OR REPRESENTATIONS OF ANY KIND WHATSOEVER EITHER EXPRESS OR
// IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF DESIGN OR
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EACH OF WHICH ARE
// SPECIFICALLY DISCLAIMED, NOR ANY WARRANTY OR REPRESENTATIONS THAT THE
// SOFTWARE IS ERROR FREE OR THAT THE SOFTWARE WILL NOT INFRINGE ANY
// PATENT, COPYRIGHT, TRADEMARK, OR OTHER THIRD PARTY PROPRIETARY RIGHTS.
// IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY OR CAUSE OF ACTION, WHETHER IN CONTRACT,
// STRICT LIABILITY, TORT, NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE. ANY RECIPIENT OR USER OF THIS SOFTWARE ACKNOWLEDGES THE
// FOREGOING, AND ACCEPTS ALL RISKS AND LIABILITIES THAT MAY ARISE FROM
// THEIR USE OF THE SOFTWARE.
// ---------------------------------

/// \file test/filters.cpp
/// \author Mark Howison

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <cutil_inline.h>
#include <cutil_gl_inline.h>

#include "TiffImage.h"
#include "Filter.hpp"
#include "SobelFilter.hpp"

#define TESTFILE "noisy-taj-mahal"

using namespace std;
using namespace xromm;

float* gpuInput;
float* gpuOutput;


int main(int argc, char** argv)
{
    TIFFSetWarningHandler(0);
    TIFF* tif = TIFFOpen(TESTFILE ".tiff", "r");
    if (!tif) {
        throw runtime_error("Unable to open test image: " TESTFILE);
    }

    TiffImage img;
    tiffImageReadMeta(tif, &img);

    if (img.samplesPerPixel != 1 || img.bitsPerSample != 8) {
        throw runtime_error("Unsupported image format");
    }

	size_t npixels = img.width * img.height;

	/* allocated buffers */
    unsigned char* input = new unsigned char[npixels];
	float* fInput = new float[npixels];
	float* fOutput = new float[npixels];
	unsigned char* output = new unsigned char[npixels];

	/* load image */
	if (tiffImageRead(tif, &img) != 1) {
		throw runtime_error("Unable to read image");
	}
    memcpy(input, img.data, npixels);
    TIFFClose(tif);

	/* convert to float */
	FILE* inputLog = fopen(TESTFILE ".in.log", "w");
	float scale = 1.f / 255.f;
	for (size_t i=0; i<npixels; i++) {
		fInput[i] = (float)input[i] * scale;
		fprintf(inputLog, "%f\n", fInput[i]);
	}
	fclose(inputLog);

	cuda::SobelFilter* filter;

	cutilSafeCall(cudaMalloc((void**)&gpuInput, npixels*sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&gpuOutput, npixels*sizeof(float)));

	cutilSafeCall(cudaMemcpy(gpuInput, fInput, npixels*sizeof(float),
				cudaMemcpyHostToDevice));

	filter = new cuda::SobelFilter();
	//filter->setScale(1.f);
	//filter->setBlend(1.f);
	filter->apply(gpuInput, gpuOutput, img.width, img.height);
	delete filter;

	cutilSafeCall(cudaThreadSynchronize());
	cutilSafeCall(cudaGetLastError());

	cutilSafeCall(cudaMemcpy(fOutput, gpuOutput, npixels*sizeof(float),
				cudaMemcpyDeviceToHost));

	/* convert to char */	
	FILE* outputLog = fopen(TESTFILE ".out.log", "w");
	for (size_t i=0; i<npixels; i++) {
		output[i] = (unsigned char)(fOutput[i] * 255.f);
		fprintf(outputLog, "%f\n", fOutput[i]);
	}
	fclose(outputLog);

    tif = TIFFOpen(TESTFILE ".sobel.tiff", "w");
    if (!tif) {
        throw runtime_error("Unable to open test image: " TESTFILE);
    }

#if 0
	TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, img.width);
	TIFFSetField(tif, TIFFTAG_IMAGELENGTH, img.height);
	TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
	TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 8);

	tsize_t linebytes = width;
	unsigned char *buf = new unsigned char[TIFFScanlineSize(tif)];
	TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif, width));

	for (size_t row=0; row < height; row++) {
		TIFFWriteScanline(tif, 
	}
	delete buf;
#endif
    memcpy(img.data, output, npixels);
	tiffImageWrite(tif, &img);
	TIFFClose(tif);

	delete input;
	delete fInput;
	delete fOutput;
	delete output;

	tiffImageFree(&img);
}

