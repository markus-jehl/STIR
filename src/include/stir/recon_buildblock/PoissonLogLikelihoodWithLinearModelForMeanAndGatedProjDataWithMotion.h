/*
  Copyright (C) 2006- 2009, Hammersmith Imanet Ltd
  Copyright (C) 2010- 2013, King's College London
  Copyright (C) 2018, University College London

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Declaration of class stir::PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  \author Sanida Mustafovic

*/

#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion_H__
#include "stir/shared_ptr.h"
#include "stir/RegisteredObject.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/ParseAndCreateFrom.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/VectorWithOffset.h"
#include "stir/GatedProjData.h"
#include "stir/GatedDiscretisedDensity.h"
#include "stir/spatial_transformation/GatedSpatialTransformation.h"

START_NAMESPACE_STIR

/*!
  \ingroup GeneralisedObjectiveFunction
  \brief a base class for LogLikelihood of independent Poisson variables
  where the mean values are linear combinations of the gated images.

 \f[
 \begin{array}{lcl}
 \Lambda_{\nu}^{(s+1)}&&=\Lambda_{\nu}^{(s)} \frac{1}{ \sum\limits_{b\in S_{l}, g} \sum\limits_{\nu'} \hat{W}^{-1}
_{\nu'g\rightarrow \nu}P_{\nu' b}A_{bg}+\beta \nabla_{\Lambda_{\nu}} E_{\nu}^{(s)}}\\
 &&\times \sum\limits_{b\in S_{l}, g} \sum\limits_{\nu'}\left(\hat{W}^{-1} _{\nu'g\rightarrow \nu}P_{\nu'
b}\frac{Y_{bg}}{\sum\limits_{\tilde{\nu}}P_{b\tilde{\nu}}\sum\limits_{\tilde{\nu}'}\hat{W} _{\tilde{\nu}'\rightarrow
\tilde{\nu}g}\Lambda_{\tilde{\nu}'}^{(s)}+\frac{B_{bg}}{A_{bg}}}\right) \end{array} \f] \par Parameters for parsing

 For more information: Tsoumpas et al (2013) Physics in Medicine and Biology

*/

template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion
    : public RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>,
                                     GeneralisedObjectiveFunction<TargetT>,
                                     PoissonLogLikelihoodWithLinearModelForMean<TargetT>>
{
private:
  typedef RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>,
                                  GeneralisedObjectiveFunction<TargetT>,
                                  PoissonLogLikelihoodWithLinearModelForMean<TargetT>>
      base_type;
  typedef PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3, float>> SingleGateObjFunc;
  VectorWithOffset<SingleGateObjFunc> _single_gate_obj_funcs;

  TimeGateDefinitions _time_gate_definitions;

public:
  //! Name which will be used when parsing a GeneralisedObjectiveFunction object
  static const char* const registered_name;

  PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion();

  //! Returns a pointer to a newly allocated target object (with 0 data).
  /*! Dimensions etc are set from the \a gated_proj_data_sptr and other information set by parsing,
    such as \c zoom, \c output_image_size_z etc.
  */
  TargetT* construct_target_ptr() const override;

  void actual_compute_subset_gradient_without_penalty(TargetT& gradient,
                                                      const TargetT& current_estimate,
                                                      const int subset_num,
                                                      const bool add_sensitivity) override;

  double actual_compute_objective_function_without_penalty(const TargetT& current_estimate, const int subset_num) override;

  Succeeded set_up_before_sensitivity(shared_ptr<const TargetT> const& target_sptr) override;

  //! Add subset sensitivity to existing data
  void add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const override;

  Succeeded actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
                                                                                   const TargetT& input,
                                                                                   const int subset_num) const override;
  Succeeded actual_accumulate_sub_Hessian_times_input_without_penalty(TargetT& output,
                                                                      const TargetT& current_image_estimate,
                                                                      const TargetT& input,
                                                                      const int subset_num) const override;

  void set_time_gate_definitions(const TimeGateDefinitions& time_gate_definitions);

  /*! \name Functions to get parameters
    \warning Be careful with changing shared pointers. If you modify the objects in
    one place, all objects that use the shared pointer will be affected.
  */
  //@{
  const GatedProjData& get_gated_proj_data() const;
  const shared_ptr<GatedProjData>& get_gated_proj_data_sptr() const;
  const int get_max_segment_num_to_process() const;
  const bool get_zero_seg0_end_planes() const;
  const GatedProjData& get_additive_gated_proj_data() const;
  const shared_ptr<GatedProjData>& get_additive_gated_proj_data_sptr() const;
  const GatedProjData& get_normalisation_gated_proj_data() const;
  const shared_ptr<GatedProjData>& get_normalisation_gated_proj_data_sptr() const;
  const ProjectorByBinPair& get_projector_pair() const;
  const shared_ptr<ProjectorByBinPair>& get_projector_pair_sptr() const;
  //@}

  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
    \warning After using any of these, you have to call set_up().
    \warning Be careful with setting shared pointers. If you modify the objects in
    one place, all objects that use the shared pointer will be affected.
  */
  //@{
  void set_recompute_sensitivity(const bool);
  void set_sensitivity_sptr(const shared_ptr<TargetT>&);
  int set_num_subsets(const int num_subsets) override;

  void set_normalisation_sptr(const shared_ptr<BinNormalisation>&) override;
  void set_additive_proj_data_sptr(const shared_ptr<ExamData>&) override;

  void set_input_data(const shared_ptr<ExamData>&) override;
  const GatedProjData& get_input_data() const override;
  //@}
protected:
  //! Filename with input projection data
  std::string _input_filename;
  std::string _motion_vectors_filename_prefix;
  std::string _reverse_motion_vectors_filename_prefix;
  std::string _gate_definitions_filename;

  //! points to the object for the total input projection data
  shared_ptr<GatedProjData> _gated_proj_data_sptr;

  //! the maximum absolute ring difference number to use in the reconstruction
  /*! convention: if -1, use get_max_segment_num()*/
  int _max_segment_num_to_process;

  const TimeGateDefinitions& get_time_gate_definitions() const;

  /**********************/
  ParseAndCreateFrom<TargetT, GatedProjData> target_parameter_parser;
  /********************************/
  //! name of file in which additive projection data are stored
  std::string _additive_gated_proj_data_filename;

  //! name of file in which normalisation projection data are stored
  std::string _normalisation_gated_proj_data_filename;

  //! points to the additive projection data
  /*! the projection data in this file is bin-wise added to forward projection results*/
  shared_ptr<GatedProjData> _additive_gated_proj_data_sptr;
  shared_ptr<GatedProjData> _normalisation_gated_proj_data_sptr;
  /*! the normalisation or/and attenuation data */
  std::string _normalisation_filename_prefix;
  //! Stores the projectors that are used for the computations
  shared_ptr<ProjectorByBinPair> _projector_pair_ptr;
  //! signals whether to zero the data in the end planes of the projection data
  bool _zero_seg0_end_planes;
  /*! the motion vectors where all information is stored */
  int _motion_correction_type; // This could be set up in a more generic way in order to choose how motion correction will be
                               // applied//
  GatedSpatialTransformation _motion_vectors;
  GatedSpatialTransformation _reverse_motion_vectors;

  //! gated image template
  GatedDiscretisedDensity _gated_image_template;
  bool actual_subsets_are_approximately_balanced(std::string& warning_message) const override;

  //! Sets defaults before parsing
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;
};

END_NAMESPACE_STIR

#endif
