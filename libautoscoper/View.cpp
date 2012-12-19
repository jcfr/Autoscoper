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

/// \file View.cpp
/// \author Andy Loomis

#include "View.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <iostream>

#include "Camera.hpp"
#include "Compositor.hpp"
#include "Filter.hpp"
#include "RayCaster.hpp"
#include "RadRenderer.hpp"

using namespace std;

namespace xromm { namespace opencl
{

View::View(Camera& camera)
{
	camera_ = &camera;
	drr_enabled = true;
	rad_enabled = true;
	drrRenderer_ = new RayCaster();
	radRenderer_ = new RadRenderer();
	maxWidth_ = 2048;
	maxHeight_ = 2048;
	drrBuffer_ = 0;
	drrFilterBuffer_ = 0;
	radBuffer_ = 0;
	radFilterBuffer_ = 0;
	filterBuffer_ = 0;
}

View::~View()
{
    delete drrRenderer_;
    delete radRenderer_;

    std::vector<Filter*>::iterator iter;
    for (iter = drrFilters_.begin(); iter != drrFilters_.end(); ++iter) {
        delete *iter;
    }

    for (iter = radFilters_.begin(); iter != radFilters_.end(); ++iter) {
        delete *iter;
    }

    delete filterBuffer_;
    delete drrBuffer_;
    delete drrFilterBuffer_;
    delete radBuffer_;
    delete radFilterBuffer_;
}

void
View::renderRad(const Buffer* buffer, unsigned width, unsigned height)
{
    init();

    if (width > maxWidth_ || height > maxHeight_) {
        cerr << "View::renderRad(): ERROR: Buffer too large." << endl;
    }
    if (width > maxWidth_) {
        width = maxWidth_;
    }
    if (height > maxHeight_) {
        height = maxHeight_;
    }

    radRenderer_->render(radBuffer_, width, height);
    filter(radFilters_, radBuffer_, buffer, width, height);
}

void
View::renderRad(GLuint pbo, unsigned width, unsigned height)
{
	GLBuffer* buffer = new GLBuffer(pbo, CL_MEM_WRITE_ONLY);

    renderRad(radFilterBuffer_, width, height);
    composite(radFilterBuffer_, radFilterBuffer_, buffer, width, height);

	delete buffer;
}

void
View::renderDrr(const Buffer* buffer, unsigned width, unsigned height)
{
    init();

    if (width > maxWidth_ || height > maxHeight_) {
        cerr << "View::renderDrr(): ERROR: Buffer too large." << endl;
    }
    if (width > maxWidth_) {
        width = maxWidth_;
    }
    if (height > maxHeight_) {
        height = maxHeight_;
    }

    drrRenderer_->render(drrBuffer_, width, height);
    filter(drrFilters_, drrBuffer_, buffer, width, height);
}

void
View::renderDrr(GLuint pbo, unsigned width, unsigned height)
{
	GLBuffer* buffer = new GLBuffer(pbo, CL_MEM_WRITE_ONLY);

    renderDrr(drrFilterBuffer_, width, height);
    composite(drrFilterBuffer_, drrFilterBuffer_, buffer, width, height);

	delete buffer;
}

void
View::render(const GLBuffer* buffer, unsigned width, unsigned height)
{
    init();

    if (drr_enabled) {
        renderDrr(drrFilterBuffer_, width, height);
    }
    else {
		drrFilterBuffer_->zero();
    }

    if (rad_enabled) {
        renderRad(radFilterBuffer_, width, height);
    }
    else {
		radFilterBuffer_->zero();
    }

    composite(drrFilterBuffer_, radFilterBuffer_, buffer, width, height);
}

void
View::render(GLuint pbo, unsigned width, unsigned height)
{
	GLBuffer* buffer = new GLBuffer(pbo, CL_MEM_WRITE_ONLY);

    render(buffer, width, height);

	delete buffer;
}

void
View::init()
{
    if (!filterBuffer_) {
        filterBuffer_    = new Buffer(maxWidth_*maxHeight_*sizeof(float));
        drrBuffer_       = new Buffer(maxWidth_*maxHeight_*sizeof(float));
        drrFilterBuffer_ = new Buffer(maxWidth_*maxHeight_*sizeof(float));
        radBuffer_       = new Buffer(maxWidth_*maxHeight_*sizeof(float));
        radFilterBuffer_ = new Buffer(maxWidth_*maxHeight_*sizeof(float));
    }
}

void
View::filter(const std::vector<Filter*>& filters,
             const Buffer* input,
             const Buffer* output,
             unsigned width,
             unsigned height)
{
    // If there are no filters simply copy the input to the output
    if (filters.size() == 0) {
		input->copy(output, width*height*sizeof(float));
        return;
    }

    // Determine which buffer will be used first so that the final
    // filter will place the results into output.
    const Buffer* buffer1;
    const Buffer* buffer2;
    if (filters.size()%2) {
        buffer1 = output;
        buffer2 = filterBuffer_;
    }
    else {
        buffer1 = filterBuffer_;
        buffer2 = output;
    }

    // Explicitly apply the first filter and altername buffers after
    vector<Filter*>::const_iterator iter = filters.begin();;

    if ((*iter)->enabled()) {
        (*iter)->apply(input, buffer1, (int)width, (int)height);
    }
    else {
		input->copy(buffer1, width*height*sizeof(float));
    }

    for (iter += 1; iter != filters.end(); ++iter) {
        if ((*iter)->enabled()) {
            (*iter)->apply(buffer1, buffer2, (int)width, (int)height);
        }
        else {
			buffer1->copy(buffer2, width*height*sizeof(float));
        }
        swap(buffer1, buffer2);
    }
}

} } // namespace xromm::opencl

