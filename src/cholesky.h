#ifndef CHOLESKY_H
#define CHOLESKY_H

#include <Eigen/Sparse>
#include "pcg.h"

namespace mschol {

template <typename T> class mt_ichol_solver;
template <typename T, typename index_t> class mt_supernodal_ichol_solver;
template <typename T, typename index_t> class mask_supernodal_ichol_solver;
template <typename T, typename index_t> class supcol_ichol_solver;
struct geom_supernode;
struct chol_level;

class ichol_precond : public preconditioner
{
 public:
  typedef supcol_ichol_solver<double, std::ptrdiff_t> solver_t;
  
  ichol_precond(const std::vector<std::shared_ptr<chol_level>> &levels, const boost::property_tree::ptree &pt);
  int analyse_pattern(const Eigen::SparseMatrix<double> &mat);
  int factorize(const Eigen::SparseMatrix<double> &mat, const bool verbose=true);
  VectorType solve(const VectorType &rhs);
  
 private:
  const std::vector<std::shared_ptr<chol_level>> &levels_;
  const boost::property_tree::ptree &pt_;
  std::vector<std::ptrdiff_t> level_ptr_;  

  std::shared_ptr<solver_t> ic_slv_;
  std::shared_ptr<geom_supernode> gs_;
  
  Eigen::SparseMatrix<double> T_;        // wavelet transformation
  Eigen::VectorXd G_;
  Eigen::PermutationMatrix<-1, -1> P_;
};

}
#endif
