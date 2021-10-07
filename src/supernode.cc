#include "supernode.h"

#include <iostream>
#include <unordered_set>
#include <numeric>
#include <set>
#include <spdlog/spdlog.h>

#include "cholesky_level.h"
#include "ichol_pattern.h"
#include "macro.h"
#include "io.h"

using namespace std;
using namespace Eigen;

namespace mschol {

geom_supernode::PermType
geom_supernode::aggregate(const std::vector<std::shared_ptr<chol_level>> &levels,
                          const VectorXd &l, const PermType &P, const PattType &SS,
                          const double rho, const index_t su_size_bound) {
  const matd_t &nods = levels.back()->nods_;
  const PermType &to_mesh = P.inverse();
  const index_t dim = SS.rows();
  const int Rd = dim/nods.cols();
  const int num_levels = levels.size();

  // [NOTICE] symmetrize sparsity pattern, additional memory cost
  spdlog::info("lower to symmetric");
  std::vector<std::vector<std::pair<index_t, index_t>>> S_col_iter(SS.rows()); 
  for (index_t j = 0; j < SS.cols(); ++j) {
    #pragma omp parallel for     
    for (index_t iter = SS.outerIndexPtr()[j]; iter < SS.outerIndexPtr()[j+1]; ++iter) {
      index_t i = SS.innerIndexPtr()[iter];
      S_col_iter[i].emplace_back(std::make_pair(j, iter));
    }
  }
  spdlog::info("----------");

  if ( !sn_ptr_.empty() ) {
    sn_ptr_.clear();
  }
  if ( !sn_ind_.empty() ) {
    sn_ind_.clear();
    sn_ind_.reserve(dim);
  }
  scale_sn_ptr_.resize(num_levels+1);

  //-> record starting and ending index of each level
  std::vector<index_t> level_ptr;
  index_t offset = 0;
  for (index_t i = levels.size()-1; i >= 0; --i) {
    level_ptr.push_back(offset);
    offset += levels[i]->W_.rows();
  }
  level_ptr.push_back(levels.back()->W_.cols());
  ASSERT(level_ptr.size() == num_levels+1);

#if 0
  //-> aggregate supernodes
  spdlog::info("group columns into supernodes");
  vector<bool> vis(dim, false);
  sn_ptr_.push_back(0);
  for (index_t p = 0; p < num_levels; ++p) {
    const index_t lev_begin = level_ptr[p], lev_end = level_ptr[p+1];

    //-> for each scale identify supernodes
    index_t num_super_nodes = 0;
    for (index_t j = lev_begin; j < lev_end; ++j) {
      if ( vis[j] ) continue;

      ++num_super_nodes;
      sn_ptr_.push_back(sn_ptr_.back());
        
      for (PattType::InnerIterator it(S, j); it; ++it) {
        const int i = it.row();
          
        if ( i < lev_begin ) continue;
        if ( i >= lev_end ) break;
        if ( vis[i] ) continue;
          
        index_t ori_i = to_mesh.indices()[i]/Rd;
        index_t ori_j = to_mesh.indices()[j]/Rd;

        double dist = zjucad::matrix::sqnorm(nods(colon(), ori_i)-nods(colon(), ori_j));
        if ( dist < rho*rho*std::min(l[ori_i]*l[ori_i],
                                     l[ori_j]*l[ori_j]) ) {
          vis[i] = true;
          sn_ind_.push_back(i); //!!!
          sn_ptr_.back() += 1;

          /// limit supernode size to reduce unimportant fill-ins
          if ( sn_ptr_.back()-sn_ptr_[sn_ptr_.size()-2] >= su_size_bound ) {
            break;
          }
        }
      }
    }

    scale_sn_ptr_[p+1] = scale_sn_ptr_[p]+num_super_nodes;
    
    // std::vector<index_t> su_num(num_super_nodes);
    // for (index_t s = scale_sn_ptr_[p], i = 0; s < scale_sn_ptr_[p+1]; ++s, ++i) {
    //   su_num[i] = sn_ptr_[s+1]-sn_ptr_[s];
    // }
    // std::reverse(su_num.begin(), su_num.end());
    // for (index_t s = scale_sn_ptr_[p], i = 0; s < scale_sn_ptr_[p+1]; ++s, ++i) {
    //   sn_ptr_[s+1] = sn_ptr_[s]+su_num[i];
    // }
    // std::reverse(&sn_ind_[lev_begin], &sn_ind_[lev_end]);
  }
#endif
  
  spdlog::info("group columns into supernodes");
  vector<bool> vis(dim, false);
  sn_ptr_.push_back(0);
  for (index_t p = 0; p < num_levels; ++p) {
    const index_t lev_begin = level_ptr[p], lev_end = level_ptr[p+1];

    //-> aggregate supernodes for each level
    index_t num_super_nodes = 0;
    for (index_t j = lev_begin; j < lev_end; ++j) {
      if ( vis[j] ) continue;

      //-> start a new supernode, for DoF j
      ++num_super_nodes;
      sn_ptr_.push_back(sn_ptr_.back());

      for (const auto &p : S_col_iter[j]) {
        index_t i = p.first; // iter = p.second;
        
        if ( i < lev_begin ) continue;
        if ( i >= lev_end ) break;
        if ( vis[i] ) continue;
          
        index_t ori_i = to_mesh.indices()[i]/Rd;
        index_t ori_j = to_mesh.indices()[j]/Rd;
        double dist = (nods.col(ori_i)-nods.col(ori_j)).squaredNorm();
        if ( dist < rho*rho*std::min(l[ori_i]*l[ori_i], l[ori_j]*l[ori_j]) ) {
          vis[i] = true;
          sn_ind_.push_back(i);
          sn_ptr_.back() += 1;

          /// limit supernode size to reduce unimportant fill-ins
          if ( sn_ptr_.back()-sn_ptr_[sn_ptr_.size()-2] >= su_size_bound ) {
            goto next_supernode;
          }
        }        
      }
      
      for (index_t iter = SS.outerIndexPtr()[j]; iter < SS.outerIndexPtr()[j+1]; ++iter) {
        const int i = SS.innerIndexPtr()[iter];
    
        if ( i < lev_begin ) continue;
        if ( i >= lev_end ) break;
        if ( vis[i] ) continue;
          
        index_t ori_i = to_mesh.indices()[i]/Rd;
        index_t ori_j = to_mesh.indices()[j]/Rd;
        double dist = (nods.col(ori_i)-nods.col(ori_j)).squaredNorm();
        if ( dist < rho*rho*std::min(l[ori_i]*l[ori_i], l[ori_j]*l[ori_j]) ) {
          vis[i] = true;
          sn_ind_.push_back(i);
          sn_ptr_.back() += 1;

          /// limit supernode size to reduce unimportant fill-ins
          if ( sn_ptr_.back()-sn_ptr_[sn_ptr_.size()-2] >= su_size_bound ) {
            goto next_supernode;
          }
        }
      }

   next_supernode: ;
    } // end of this level

    scale_sn_ptr_[p+1] = scale_sn_ptr_[p]+num_super_nodes;
  }
  ASSERT(sn_ind_.size() == dim);
  
  const int SU_NUM = sn_ptr_.size()-1;
  spdlog::info("supernode number: {}", SU_NUM);
  // index_t max_sz = 0, min_sz = std::numeric_limits<index_t>::max();
  // index_t count_su_1 = 0;
  // for (index_t i = 0; i < SU_NUM; ++i) {
  //   index_t curr_sz = sn_ptr_[i+1]-sn_ptr_[i];
  //   max_sz = std::max(curr_sz, max_sz);
  //   min_sz = std::min(curr_sz, min_sz);
  //   if ( curr_sz == 1 ) ++count_su_1;
  // }
  // spdlog::info("min max supernode size: {} {}", min_sz, max_sz);
  // spdlog::info("number of size-1 supernodes: {}", count_su_1);

  //-> reorder to group supernodes into contiguous blocks
  PermType P_su(dim);
  for (int i = 0; i < sn_ind_.size(); ++i) {
    P_su.indices()[sn_ind_[i]] = i;
  }
  std::iota(sn_ind_.begin(), sn_ind_.end(), 0);

  //-> permute original S, clustering supernodes into contiguous block
  spdlog::info("build tilde S_rho");
  {
    //-> identiy col's supernodal ID
    std::vector<index_t> col_to_su(dim);
    for (int i = 0; i < sn_ptr_.size()-1; ++i) {
      for (int cnt = sn_ptr_[i]; cnt < sn_ptr_[i+1]; ++cnt) {
        const int col_idx = sn_ind_[cnt];
        col_to_su[col_idx] = i;
      }
    }

    // spdlog::info("===");
    // std::unordered_set<index_t> vis;
    // vector<Triplet<index_t>> trips;
    // for (int j = 0; j < S.cols(); ++j) {
    //   for (PattType::InnerIterator it(S, j); it; ++it) {
    //     const index_t su_i = col_to_su[P_su.indices()[it.row()]];
    //     const index_t su_j = col_to_su[P_su.indices()[it.col()]];
    //     if ( su_i <= su_j && vis.find(su_i*SU_NUM+su_j) == vis.end() ) {
    //       vis.insert(su_i*SU_NUM+su_j);
    //       trips.emplace_back(Triplet<index_t>(su_i, su_j, 1));
    //     }
    //   }
    // }

    spdlog::info("=== S_su ===");
    std::unordered_set<index_t> vis;
    vector<Triplet<index_t>> trips;
    for (int j = 0; j < SS.cols(); ++j) {
      for (PattType::InnerIterator it(SS, j); it; ++it) {
        index_t su_i = col_to_su[P_su.indices()[it.row()]];
        index_t su_j = col_to_su[P_su.indices()[it.col()]];
        if ( su_i <= su_j && vis.find(su_i*SU_NUM+su_j) == vis.end() ) {
          vis.insert(su_i*SU_NUM+su_j);
          trips.emplace_back(Triplet<index_t>(su_i, su_j, 1));
        }

        su_i = col_to_su[P_su.indices()[it.col()]];
        su_j = col_to_su[P_su.indices()[it.row()]];
        if ( su_i <= su_j && vis.find(su_i*SU_NUM+su_j) == vis.end() ) {
          vis.insert(su_i*SU_NUM+su_j);
          trips.emplace_back(Triplet<index_t>(su_i, su_j, 1));
        }        
      }
    }
    
    S_su_.resize(SU_NUM, SU_NUM);
    S_su_.reserve(trips.size());
    S_su_.setFromTriplets(trips.begin(), trips.end());
  }

  //#define WRITE_PATTERN
#ifdef WRITE_PATTERN
  vector<Triplet<double>> trips;
  for (index_t j = 0; j < S_su_.cols(); ++j) {
    for (SparseMatrix<index_t>::InnerIterator it(S_su_, j); it; ++it) {
      const index_t I = it.row(), J = j;
      for (index_t iter_I = sn_ptr_[I]; iter_I < sn_ptr_[I+1]; ++iter_I) {
        for (index_t iter_J = sn_ptr_[J]; iter_J < sn_ptr_[J+1]; ++iter_J) {
          trips.emplace_back(Triplet<double>(sn_ind_[iter_I], sn_ind_[iter_J], 1));
        }
      }
    }
  }
  SparseMatrix<double> oldS(SS.rows(), SS.cols());
  oldS.setFromTriplets(trips.begin(), trips.end());
  spdlog::info("old S nnz: {}", oldS.nonZeros());
  write_sparse_matrix("./oldS.spmat", oldS);
#endif
    
  return P_su;
}

geom_supernode::PermType geom_supernode::multicoloring() {
  //-> in-level supernodal multicoloring
  decltype(S_su_) sym_Su;
  sym_Su = S_su_.selfadjointView<Eigen::Upper>();
  
  PermType su_colorP = adjust_inlevel_order<decltype(S_su_)::Scalar, decltype(S_su_)::StorageIndex>(sym_Su, scale_sn_ptr_, 0);
  ASSERT(su_colorP.rows() == S_su_.rows());
  {
    SparseMatrix<index_t> tmp_S;
    //    tmp_S = S_su_.selfadjointView<Eigen::Upper>().twistedBy(su_colorP);
    tmp_S = sym_Su.twistedBy(su_colorP);
    S_su_ = tmp_S.triangularView<Eigen::Upper>();
  }

  const index_t dim = sn_ind_.size();
  const index_t su_num = sn_ptr_.size()-1;

  std::vector<index_t> su_size(su_num);
  for (index_t i = 0; i < su_size.size(); ++i) {
    su_size[i] = sn_ptr_[i+1]-sn_ptr_[i];
  }

  std::vector<index_t> perm_su_size(su_size.size());
  for (index_t i = 0; i < perm_su_size.size(); ++i) {
    perm_su_size[su_colorP.indices()[i]] = su_size[i];
  }

  std::vector<index_t> perm_sn_ptr(sn_ptr_.size());
  perm_sn_ptr[0] = 0;
  for (index_t i = 0; i < perm_su_size.size(); ++i) {
    perm_sn_ptr[i+1] = perm_sn_ptr[i]+perm_su_size[i];
  }

  std::vector<index_t> perm_sn_ind(sn_ind_.size());
  for (size_t i = 0; i < su_num; ++i) {
    index_t perm_i = su_colorP.indices()[i];
      
    for (index_t iter1 = sn_ptr_[i], iter2 = perm_sn_ptr[perm_i]; iter1 < sn_ptr_[i+1]; ++iter1, ++iter2) {
      perm_sn_ind[iter2] = sn_ind_[iter1];
    }
  }

  PermType fullcolorP(dim);
  for (index_t i = 0; i < dim; ++i) {
    fullcolorP.indices()[perm_sn_ind[i]] = i;
  }
  std::iota(perm_sn_ind.begin(), perm_sn_ind.end(), 0);

  //-> reset two vectors
  sn_ptr_ = perm_sn_ptr;
  sn_ind_ = perm_sn_ind;

  return fullcolorP;
}

}
