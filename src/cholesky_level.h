#ifndef CHOLESKY_LEVEL_H
#define CHOLESKY_LEVEL_H

#include <Eigen/Sparse>
#include "types.h"

namespace mschol {

struct chol_level
{
  // prescribed refinement and its kernel to next level
  Eigen::SparseMatrix<double> C_, W_;
  // mesh
  mati_t cell_; matd_t nods_;
  std::string mesh_type_;
  Eigen::VectorXd calc_supp_scale(const double *Vol) const;
};

}
#endif
