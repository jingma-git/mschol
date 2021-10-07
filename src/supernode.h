#ifndef SUPERNODE_H
#define SUPERNODE_H

#include <memory>
#include <bitset>
#include <Eigen/Sparse>

namespace mschol {

struct chol_level;

typedef std::bitset<64> mask_type;

struct geom_supernode
{
  typedef std::ptrdiff_t index_t;
  typedef Eigen::SparseMatrix<char, Eigen::ColMajor, std::ptrdiff_t> PattType;
  typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PermType;

  //-> the column indicies of i-th supernode are stored consequtively [sn_ptr_[i], sn_ptr_[i+1])
  std::vector<index_t> sn_ptr_, sn_ind_;
  //-> starting and ending indices of supernodes on *each scale*
  std::vector<index_t> scale_sn_ptr_;
  //-> connectivity of supernodes
  Eigen::SparseMatrix<index_t> S_su_;
  Eigen::SparseMatrix<mask_type> S_MA_;
  std::vector<mask_type> mask_;

  PermType aggregate(const std::vector<std::shared_ptr<chol_level>> &levels,
                     const Eigen::VectorXd &l, const PermType &P, const PattType &SS,
                     const double rho, const index_t su_size_bound);
  PermType multicoloring();
};

}
#endif
