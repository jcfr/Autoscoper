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

/// \file Tracker.hpp
/// \author Andy Loomis

#ifndef XROMM_TRACKER_H
#define XROMM_TRACKER_H

#include <vector>
#include <string>

#include "Filter.hpp"

#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
#include "gpu/cuda/RayCaster.hpp"
#include "gpu/cuda/RadRenderer.hpp"
#include "gpu/cuda/BackgroundRenderer.hpp"
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
#include "gpu/opencl/RayCaster.hpp"
#include "gpu/opencl/RadRenderer.hpp"
#include "gpu/opencl/BackgroundRenderer.hpp"
#include "gpu/opencl/OpenCL.hpp"
#endif

#include "Trial.hpp"


namespace xromm
{
  class Camera;
  class CoordFrame;

  namespace gpu
  {
    class Filter;
    class View;
    class VolumeDescription;

  } // namespace gpu


  class Tracker
  {
  public:

    Tracker();
    ~Tracker();
    void init();
    void load(const Trial& trial);
    Trial* trial() { return &trial_; }
    void optimize(int frame, int dframe, int repeats, int opt_method, unsigned int max_iter, double min_limit, double max_limit, int cf_model, unsigned int max_stall_iter);
    double minimizationFunc(const double* values) const;
    std::vector <double> trackFrame(unsigned int volumeID, double* xyzpr) const;
    std::vector<gpu::View*>& views() { return views_; }
    const std::vector<gpu::View*>& views() const { return views_; }
    gpu::View* view(size_t i) { return views_.at(i); }
    const gpu::View* view(size_t i) const { return views_.at(i); }
    void updateBackground();
    void setBackgroundThreshold(float threshold);
    std::vector<unsigned char> getImageData(unsigned volumeID, unsigned camera, double* xyzpr, unsigned& width, unsigned& height);


    // Bardiya Cost Function for Implants
    //double implantMinFunc(const double* values) const;
    //std::vector<double> trackImplantFrame(unsigned int volumeID, double * xyzypr) const;

    void getFullDRR(unsigned int volumeID) const;


  private:
    void calculate_viewport(const CoordFrame& modelview, double* viewport) const;

    int optimization_method = (int)0;
    int cf_model_select = (int)0;
    Trial trial_;
    std::vector <gpu::VolumeDescription*> volumeDescription_;
    std::vector <gpu::View*> views_;
#if defined(Autoscoper_RENDERING_USE_CUDA_BACKEND)
    Buffer* rendered_drr_;
    Buffer* rendered_rad_;
    Buffer* background_mask_;
    Buffer* drr_mask_;
#elif defined(Autoscoper_RENDERING_USE_OpenCL_BACKEND)
    gpu::Buffer* rendered_drr_;
    gpu::Buffer* rendered_rad_;
    gpu::Buffer* background_mask_;
    gpu::Buffer* drr_mask_;
#endif
  };

} // namespace XROMM

#endif // XROMM_TRACKER_H
