//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
    Copyright (C) 2023, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup projdata

  \brief Declaration of class stir::Sinogram

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/
#ifndef __Sinogram_h__
#define __Sinogram_h__

#include "stir/Array.h"
#include "stir/ProjDataInfo.h"
#include "stir/SinogramIndices.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

/*!
  \ingroup projdata
  \brief A class for 2d projection data.

  This represents a subset of the full projection. SegmentIndices and axial_pos_num
  are fixed.

*/
template <typename elemT>
class Sinogram : public Array<2, elemT>
{
private:
  typedef Array<2, elemT> base_type;
#ifdef SWIG
  // SWIG needs the next typedef to be public
public:
#endif
  typedef Sinogram<elemT> self_type;
#ifdef SWIG
  // SWIG needs a default constructor
  inline Sinogram() {}
#endif

public:
  //! Construct sinogram from proj_data_info pointe and indices.  Data are set to 0.
  inline Sinogram(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr, const SinogramIndices&);

  //! Construct sinogram with data set to the array.
  inline Sinogram(const Array<2, elemT>& p, const shared_ptr<const ProjDataInfo>& proj_data_info_sptr, const SinogramIndices&);

  //! Construct sinogram from proj_data_info pointer, axial position and segment number.  Data are set to 0.
  /*!
    \deprecated Use version with SinogramIndices instead.
  */
  inline Sinogram(const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
                  const int ax_pos_num,
                  const int segment_num,
                  const int timing_pos_num = 0);

  //! Construct sinogram with data set to the array.
  /*!
    \deprecated Use version with SinogramIndices instead.
  */
  inline Sinogram(const Array<2, elemT>& p,
                  const shared_ptr<const ProjDataInfo>& proj_data_info_ptr,
                  const int ax_pos_num,
                  const int segment_num,
                  const int timing_pos_num = 0);

  //! Get indices
  inline SinogramIndices get_sinogram_indices() const;
  //! Get segment number
  inline int get_segment_num() const;
  //! Get number of axial positions
  inline int get_axial_pos_num() const;
  //! Get timing position index
  inline int get_timing_pos_num() const;
  //! Get minimum view number
  inline int get_min_view_num() const;
  //! Get maximum view number
  inline int get_max_view_num() const;
  //! Get number of views
  inline int get_num_views() const;
  //! Get minimum number of tangetial positions
  inline int get_min_tangential_pos_num() const;
  //! Get maximum number of tangential positions
  inline int get_max_tangential_pos_num() const;
  //! Get number of tangential positions
  inline int get_num_tangential_poss() const;

  //! Get an empty sinogram of the same dimensions, segment_num etc.
  inline Sinogram get_empty_copy(void) const;

  //! Overloading Array::grow
  void grow(const IndexRange<2>& range) override;
  //! Overloading Array::resize
  void resize(const IndexRange<2>& range) override;

  //! Get shared pointer to proj data info
  inline shared_ptr<const ProjDataInfo> get_proj_data_info_sptr() const;

  // inline Sinogram operator = (const Sinogram &s) const;

  //! \name Equality
  //@{
  //! Checks if the 2 objects have the proj_data_info, segment_num etc.
  /*! If they do \c not have the same characteristics, the string \a explanation
      explains why.
  */
  bool has_same_characteristics(self_type const&, std::string& explanation) const;

  //! Checks if the 2 objects have the proj_data_info, segment_num etc.
  /*! Use this version if you do not need to know why they do not match.
   */
  bool has_same_characteristics(self_type const&) const;

  //! check equality (data has to be identical)
  /*! Uses has_same_characteristics() and Array::operator==.
      \warning This function uses \c ==, which might not be what you
      need to check when \c elemT has data with float or double numbers.
  */
  bool operator==(const self_type&) const;

  //! negation of operator==
  bool operator!=(const self_type&) const;
  //@}

private:
  shared_ptr<const ProjDataInfo> proj_data_info_ptr;
  SinogramIndices _indices;
};

END_NAMESPACE_STIR

#include "stir/Sinogram.inl"

#endif
