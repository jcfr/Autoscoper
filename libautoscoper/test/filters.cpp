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

#include <cstdio>
#include <string>
#include <iostream>
#include <stdexcept>
#include <GLUT/glut.h>

#include "TiffImage.h"
#include "SobelFilter.hpp"
#include "ContrastFilter.hpp"
#include "SharpenFilter.hpp"
#include "GaussianFilter.hpp"

#define TESTFILE "noisy-taj-mahal"

using namespace std;
using namespace xromm;

static TiffImage img;
static size_t npixels;

static unsigned char* input;
static unsigned char* output;

static float* fInput;
static float* fOutput;

static opencl::Buffer* clInput;
static opencl::Buffer* clOutput;

void writeOutput(const char* name)
{
	clOutput->write((void*)fOutput);
	
	/* convert to char */	
	string filename(TESTFILE ".");
	filename.append(name);
	filename.append(".txt");
	FILE* outputLog = fopen(filename.c_str(), "w");
	for (size_t i=0; i<npixels; i++) {
		output[i] = (unsigned char)(fOutput[i] * 255.f);
		fprintf(outputLog, "%f\n", fOutput[i]);
	}
	fclose(outputLog);

	filename.assign(TESTFILE ".");
	filename.append(name);
	filename.append(".tiff");
    TIFF* tif = TIFFOpen(filename.c_str(), "w");
    if (!tif) {
        throw runtime_error("Unable to open test image: " TESTFILE);
    }

    memcpy(img.data, output, npixels);
	tiffImageWrite(tif, &img);
	TIFFClose(tif);
}

void testSobel()
{
	opencl::SobelFilter* filter = new opencl::SobelFilter();
	clInput->read((const void*)fInput);
	filter->apply(clInput, clOutput, img.width, img.height);
	writeOutput("sobel");
	delete filter;
}

void testContrast()
{
	opencl::ContrastFilter* filter = new opencl::ContrastFilter();
	clInput->read((const void*)fInput);
	filter->apply(clInput, clOutput, img.width, img.height);
	writeOutput("contrast");
	delete filter;
}

void testSharpen()
{
	opencl::SharpenFilter* filter = new opencl::SharpenFilter();
	clInput->read((const void*)fInput);
	filter->apply(clInput, clOutput, img.width, img.height);
	writeOutput("sharpen");
	delete filter;
}

void testGaussian()
{
	opencl::GaussianFilter* filter = new opencl::GaussianFilter();
	clInput->read((const void*)fInput);
	filter->apply(clInput, clOutput, img.width, img.height);
	writeOutput("gaussian");
	delete filter;
}

int main(int argc, char** argv)
{
	glutInit(&argc, (char**)argv);
	glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
	glutCreateWindow( "OpenCL <-> OpenGL Test" );

	TIFFSetWarningHandler(0);
	TIFF* tif = TIFFOpen(TESTFILE ".tiff", "r");
	if (!tif) {
		throw runtime_error("Unable to open test image: " TESTFILE);
	}

	tiffImageReadMeta(tif, &img);
	tiffImageDumpMeta(&img);

	if (img.samplesPerPixel != 1 || img.bitsPerSample != 8) {
		throw runtime_error("Unsupported image format");
	}

	cout << "Image dimensions: " << img.width << " x " << img.height << "\n";
	npixels = img.width * img.height;
	cout << "Image size: " << npixels << endl;

	/* allocated buffers */
	input = new unsigned char[npixels];
	fInput = new float[npixels];
	fOutput = new float[npixels];
	output = new unsigned char[npixels];

	/* load image */
	if (tiffImageRead(tif, &img) != 1) {
		throw runtime_error("Unable to read image");
	}
	memcpy(input, img.data, npixels);
	TIFFClose(tif);

	/* convert to float */
	FILE* inputLog = fopen(TESTFILE ".txt", "w");
	float scale = 1.f / 255.f;
	for (size_t i=0; i<npixels; i++) {
		fInput[i] = (float)input[i] * scale;
		fprintf(inputLog, "%f\n", fInput[i]);
	}
	fclose(inputLog);

	clInput = new opencl::Buffer(npixels*sizeof(float), CL_MEM_READ_ONLY);
	clOutput = new opencl::Buffer(npixels*sizeof(float), CL_MEM_WRITE_ONLY);

	testSobel();
	testContrast();
	testSharpen();
	testGaussian();

	delete input;
	delete output;

	delete fInput;
	delete fOutput;

	delete clInput;
	delete clOutput;

	tiffImageFree(&img);
}

