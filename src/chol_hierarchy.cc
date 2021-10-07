#include "chol_hierarchy.h"
#include <spdlog/spdlog.h>
#include "macro.h"
#include "volume.h"
#include "io.h"
#include "cholesky_level.h"

using namespace std;

namespace mschol {

chol_hierarchy::chol_hierarchy(const mati_t &tris, const matd_t &nods, const std::string &mesh_type)
    : tris_(tris), nods_(nods), mesh_type_(mesh_type) {
  //-> build kd-tree
  pts_.resize(nods.cols(), nods.rows());
  std::copy(nods.data(), nods.data()+nods.size(), pts_.data());
  kdt_ = std::make_shared<kd_tree_t>(pts_.cols(), pts_, 10);
  kdt_->index->buildIndex();
}

void chol_hierarchy::build(std::vector<std::shared_ptr<chol_level>> &levels, const size_t num_coarse_node, const size_t prb_rd, Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> *P) {
  //-> 2d or 3d problems
  const size_t rd = nods_.rows();
  ASSERT(rd == 2 || rd == 3);
  const double fac = (rd == 2) ? 0.25 : 0.125;
  spdlog::info("mesh dimension: {}", rd);
  spdlog::info("problem dimension: {}", prb_rd);
    
  const size_t N = nods_.cols();
  spdlog::info("total node number N={}", N);

  std::vector<size_t> num_dof;
  size_t node_num_per_level = N;
  num_dof.push_back(node_num_per_level);
  while ( node_num_per_level > num_coarse_node ) {
    node_num_per_level = static_cast<size_t>(fac*node_num_per_level);
    num_dof.push_back(node_num_per_level);
  }
  std::reverse(num_dof.begin(), num_dof.end());
  std::cout << "--- number of nodes on each level ---" << std::endl;
  print_vector(num_dof);
    
  const int num_levels = num_dof.size();
  levels.resize(num_levels);
  for (auto &p : levels) {
    p = std::make_shared<chol_level>();
  }
  spdlog::info("total number of levels: {}", num_levels);    
    
  std::vector<double> len_scale(num_levels);
  {
    volume_calculator vc(tris_, nods_, mesh_type_);
    const double VOLUME = vc.compute();
    
    for (int i = 0; i < len_scale.size(); ++i) {
      if ( rd == 2 ) {
        len_scale[i] = sqrt(2.0)*sqrt(VOLUME/num_dof[i]);
      } else if ( rd == 3 ) {
        len_scale[i] = sqrt(3.0)*cbrt(VOLUME/num_dof[i]);
      }
    }
    std::cout << "--- mean length scale on each level ---" << std::endl;
    print_vector(len_scale);
  }
    
  heap_t H;
  std::vector<handle_t> han(N);
    
  size_t count = 0;
  Eigen::VectorXi seq(N);
  std::vector<double> l(N);
  std::vector<bool> valid(N, true);

  const size_t START = 0;
  seq[count++] = START;
  l[START] = 0;    
  valid[START] = false;
  for (size_t j = 0; j < N; ++j) {
    if ( valid[j] ) {
      l[j] = (nods_.col(j)-nods_.col(START)).squaredNorm();
      han[j] = H.push(std::make_pair(j, l[j]));
    }
  }

  std::vector<size_t>::iterator ptr_dof = num_dof.begin();
  std::vector<double>::iterator ptr_len = len_scale.begin();
  while ( !H.empty() ) {
    size_t pid; double radius;
    std::tie(pid, radius) = H.top();
    H.pop();

    seq[count] = pid;
    l[pid] = 0;
    valid[pid] = false;

    // relax for remaining nodes, radius search O(log N)
    std::vector<std::pair<long int, double>> res;
    if ( ptr_len != len_scale.end() ) {
      kdt_->index->radiusSearch(&nods_(0, pid), (*ptr_len)*(*ptr_len), res, nanoflann::SearchParams());
    }
    for (const auto p : res) {
      const size_t j = p.first;
      if ( valid[j] ) {
        double dij = (nods_.col(j)-nods_.col(pid)).squaredNorm();
        if ( dij < l[j] ) {
          l[j] = dij;
          H.decrease(han[j], std::make_pair(j, l[j]));
        }
      }
    }

    ++count;
    if ( count >= *ptr_dof ) { // step into a new level
      ++ptr_dof;
      ++ptr_len;
    }
  }
  ASSERT(count == N);
  ASSERT(ptr_dof == num_dof.end());
  ASSERT(ptr_len == len_scale.end());

  veci_t remap(N);
  for (size_t i = 0; i < N; ++i) {
    remap[seq[i]] = i;
  }
  if ( P ) {
    P->resize(N);
    for (size_t i = 0; i < seq.size(); ++i) {
      P->indices()[seq[i]] = i;
    }
  }

  //-> finest mesh
  levels.back()->nods_.resize(nods_.rows(), nods_.cols());
  for (size_t i = 0; i < nods_.cols(); ++i) {
    levels.back()->nods_.col(i) = nods_.col(seq[i]);
  }
  levels.back()->cell_.resize(tris_.rows(), tris_.cols());
  for (size_t i = 0; i < levels.back()->cell_.size(); ++i) {
    auto ptr_curr = levels.back()->cell_.data()+i;
    auto ptr_orig = tris_.data()+i;
    *ptr_curr = remap[*ptr_orig];
  }
  levels.back()->mesh_type_ = mesh_type_;

  // remaining level infos
  for (int i = levels.size()-1; i >= 1; --i) {
    auto &curr = levels[i];
    auto &prev = levels[i-1];

    size_t num_h = num_dof[i], num_H = num_dof[i-1];
    ASSERT(num_h == curr->nods_.cols());
            
    prev->mesh_type_ = mesh_type_;
    prev->nods_ = curr->nods_.leftCols(num_H);

    Eigen::SparseMatrix<double> Id(prb_rd*num_h, prb_rd*num_h);
    Id.setIdentity();      
    curr->C_ = Id.topRows(prb_rd*num_H);
    curr->W_ = Id.bottomRows(prb_rd*(num_h-num_H));
  }

  //-> coarsest level
  levels.front()->C_.resize(prb_rd*num_dof[0], prb_rd*num_dof[0]);
  levels.front()->W_.resize(0, prb_rd*num_dof[0]);
  levels.front()->C_.setIdentity();
  levels.front()->W_.setZero();
}  

}
