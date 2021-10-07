#ifndef CHOL_HIERARCHY_H
#define CHOL_HIERARCHY_H

#include <Eigen/Sparse>
#include <boost/heap/binomial_heap.hpp>
#include <memory>
#include "nanoflann.hpp"
#include "types.h"

namespace mschol {

struct chol_level;

class chol_hierarchy
{
 public:
  typedef std::pair<size_t, double> heap_data_t;
  struct heap_data_comp {
    bool operator ()(const heap_data_t &lhs, const heap_data_t &rhs) const {
      return lhs.second < rhs.second;
    }
  };
  typedef boost::heap::binomial_heap<heap_data_t, boost::heap::compare<heap_data_comp>> heap_t;
  typedef heap_t::handle_type handle_t;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenCSR;
  typedef nanoflann::KDTreeEigenMatrixAdaptor<EigenCSR> kd_tree_t;

  chol_hierarchy(const mati_t &tris, const matd_t &nods, const std::string &mesh_type);
  void build(std::vector<std::shared_ptr<chol_level>> &levels, const size_t num_coarse_node, const size_t prb_rd, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> *P=nullptr);  

 private:
  const mati_t &tris_;
  const matd_t &nods_;
  const std::string mesh_type_;

  std::shared_ptr<kd_tree_t> kdt_;
  EigenCSR pts_;
};

}
#endif
