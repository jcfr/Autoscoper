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

/// \file test/render.cpp
/// \author Mark Howison

#include <cstdio>
#include <cstring>
#include <string>
#include <iostream>
#include <stdexcept>

#define GL_GLEXT_PROTOTYPES 1
#include <GL/glew.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "OpenCL.hpp"
#include "Volume.hpp"
#include "VolumeDescription.hpp"
#include "RayCaster.hpp"
#include "TiffImage.h"

#define TESTFILE "XMC3_13735"

using namespace std;
using namespace xromm;

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
	glutCreateWindow( "OpenCL <-> OpenGL Test" );

	cerr << "Initializing OpenCL-OpenGL interoperability..." << endl;
	opencl::opencl_global_gl_context();

	const size_t w = 1024;
	const size_t h = 1024;

	Volume vol(TESTFILE ".tif");
	vol.flipX(true);
	vol.flipY(true);

	cout << "Volume dimensions: " << vol.width() << " x " << vol.height() << " x " << vol.depth() << " (" << vol.bps() << " bps)\n";

	opencl::VolumeDescription vd(vol);
	opencl::RayCaster rc;
	opencl::Buffer* buf = new opencl::Buffer(sizeof(float)*w*h);

	rc.setVolume(vd);
	double view[16] = {
		0.707107, -0.405580, 0.579228, 250.000000,
        0.000000, 0.819152, 0.573576, 250.000000,
       -0.707107, -0.405580, 0.579228, 250.000000,
        0.000000, 0.000000, 0.000000, 1.0
	};

	double view2[16] = { 0.772409, 0.576666, -0.266161, -114.097438,
		  -0.596309, 0.514191, -0.616460, -265.904840,
		   -0.218634, 0.634873, 0.741037, 322.143432,
		    0.000000, 0.000000, 0.000000, 1.000000};

	double view3[16] =  { 0.885746, 0.090143, 0.455332, 1004.191210,
		  -0.353100, 0.767583, 0.534917, 1181.029445,
		   -0.301286, -0.634579, 0.711714, 1568.067008,
		    0.000000, 0.000000, 0.000000, 1.00000};
	rc.setInvModelView(view);
	rc.setViewport(-1.173121, -1.000000, 2.346241, 2.000000);
	rc.setRayIntensity(10.f);
	//rc.setSampleDistance(0.1f);
	rc.render(buf, w, h);

	cout << "ray intensity: " << rc.getRayIntensity() << endl;

	TiffImage img;
	img.width = w;
	img.height = h;
	img.bitsPerSample = 8;
	img.photometric = 1;
	img.orientation = 1;
	img.samplesPerPixel = 1;
	img.planarConfig = 1;
	img.sampleFormat = 1;
	img.compression = 1;
	img.dataSize = w*h;
	img.data = malloc(img.dataSize);

	tiffImageDumpMeta(&img);

	float* buf_cpu = new float[w*h];
	buf->write(buf_cpu); 
	delete buf;

	float m = 0.f;
	for (size_t i=0; i<w*h; i++) m = max(m, buf_cpu[i]);
	cout << "max pixel value: " << m << endl;
	float norm = 255.f / m;

	unsigned char* data = (unsigned char*)img.data;
	FILE* outputLog = fopen(TESTFILE ".render.txt", "w");
	for (size_t i=0; i<w*h; i++) {
		data[i] = (unsigned char)(norm * buf_cpu[i]);
		fprintf(outputLog, "%f\n", buf_cpu[i]);
	}
	delete buf_cpu;
	fclose(outputLog);

	TIFF* tif = TIFFOpen(TESTFILE ".render.tif", "w");
	tiffImageWrite(tif, &img);
	TIFFClose(tif);

	free(img.data);
}

