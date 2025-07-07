//
//
/*!
  \file
  \ingroup Parallelproj

  \brief Defines stir::detail::ParallelprojHelper

  \author Kris Thielemans

*/
/*
    Copyright (C) 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_recon_buildblock_ParallelprojHelper_h__
#define __stir_recon_buildblock_ParallelprojHelper_h__

#include "stir/common.h"
#include <vector>
#include <array>

START_NAMESPACE_STIR

template <int num_dimensions, class elemT>
class DiscretisedDensity;
class ProjDataInfo;

namespace detail
{
/*!
  \ingroup projection
  \ingroup Parallelproj
  \brief Helper class for Parallelproj's projectors
*/
class ParallelprojHelper
{
public:
  ~ParallelprojHelper();
  ParallelprojHelper(const ProjDataInfo& p_info, const DiscretisedDensity<3, float>& density);

  // parallelproj arrays
  std::array<float, 3> voxsize;
  std::array<int, 3> imgdim;
  std::array<float, 3> origin;
  float* xstart;
  float* xend;

  long long num_image_voxel;
  long long num_lors;

  float sigma_tof;
  float tofcenter_offset;
  float tofbin_width;
  short num_tof_bins;
};

} // namespace detail

END_NAMESPACE_STIR

#ifdef __cplusplus
extern "C" {
#endif

/** @brief copy a float array to all visible cuda devices
 *
 *  The number of visible cuda devices is determined automatically via the CUDA
 * API
 *
 *  @param    h_array   array of shape [n] on the host
 *  @param    n         number of array elements
 *  @return   a pointer to all devices arrays
 */
float **copy_float_array_to_all_devices(const float *h_array, long long n);

/** @brief free device array on all visible cuda devices
 *
 *  The number of visible cuda devices is determined automatically via the CUDA
 * API
 *
 *  @param d_array a pointer to all devices arrays
 */
void free_float_array_on_all_devices(float **d_array);

/** @brief sum multiple versions of an array on different devices on first
 * device
 *
 *  The number of visible cuda devices is determined automatically via the CUDA
 * API This becomes usefule when multiple devices backproject into separate
 * images.
 *
 *  @param d_array a pointer to all devices arrays
 *  @param    n         number of array elements
 */
void sum_float_arrays_on_first_device(float **d_array, long long n);

/** @brief copy a (summed) float array from first device back to host
 *
 *  The number of visible cuda devices is determined automatically via the CUDA
 * API
 *
 *  @param  d_array   a pointer to all devices arrays of shape [n]
 *  @param  n         number of array elements
 *  @param  i_dev     device number
 *  @param  h_array   array of shape [n] on the host used for output
 */
void get_float_array_from_device(float **d_array, long long n, int i_dev,
                                 float *h_array);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // __stir_recon_buildblock_ParallelprojHelper_h__
