#ifndef ICHOL_ORDER_PATTERN_H
#define ICHOL_ORDER_PATTERN_H

#include <Eigen/Sparse>
#include <unordered_set>
#include <boost/property_tree/ptree.hpp>
#include <spdlog/spdlog.h>

#include "volume.h"
#include "nanoflann.hpp"
#include "cholesky_level.h"

namespace mschol {

struct chol_level;

template <typename val_t>
class ichol_order_patt_base
{
 public:
  typedef Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> PermutationType;
  typedef Eigen::SparseMatrix<double, Eigen::ColMajor> CSCType;
  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> CSRType;
  typedef Eigen::SparseMatrix<val_t, Eigen::ColMajor, std::ptrdiff_t> PatternType;

  ichol_order_patt_base(const std::vector<std::shared_ptr<chol_level>> &levels, const char basis_type)
      : levels_(levels), nods_(levels.back()->nods_),
        Rd_(levels.back()->C_.cols()/nods_.cols()),
        basis_type_(basis_type) {
    spdlog::info("# patt base Rd: {}", Rd_);
    spdlog::info("# patt base nods.cols(): {}", nods_.cols());
    const size_t dim = levels.back()->C_.cols();
    T_.resize(dim, dim);
    T_.setIdentity();
  }

  virtual ~ichol_order_patt_base() {}
  
  virtual void run(const double rho) = 0;

  //-> all following read-only funcs can be only called after run()
  inline const CSRType&          getT() const { return T_; }
  inline const PermutationType&  getFullP() const { return fullP_; }
  inline const PatternType&      getZeroS() const { return zeroS_; }
  inline const Eigen::VectorXd&  getL() const { return l_; }

 protected:
  friend struct geom_supernode;
  
  const std::vector<std::shared_ptr<chol_level>> &levels_;
  const matd_t &nods_;  
  const size_t Rd_;              //-> dimension
  const char basis_type_;

  Eigen::VectorXd l_;            //-> with size of nods.cols()
  matd_t center_;                //-> center of basis support  
  
  PermutationType fullP_;        //-> record permutation of nodes
  CSRType T_;                    //-> wavelet transformation
  PatternType zeroS_;            //-> sparsity pattern
};

template <typename val_t>
class ichol_ftoc_order : public ichol_order_patt_base<val_t>
{
 public:
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenCSR;
  typedef nanoflann::KDTreeEigenMatrixAdaptor<EigenCSR> kd_tree_t;
  
  ichol_ftoc_order(const std::vector<std::shared_ptr<chol_level>> &levels, const char basis_type)
      : ichol_order_patt_base<val_t>(levels, basis_type) {    
    spdlog::info("Rd: {}", ichol_order_patt_base<val_t>::Rd_);
    
    //-> [l] is the length scale of basis support
    const size_t N = ichol_order_patt_base<val_t>::nods_.cols();
    auto &L = ichol_order_patt_base<val_t>::l_;
    L.setZero(N);

    //-> get volume of the finest mesh
    const auto &FL = levels.back();
    volume_calculator vc(FL->cell_, FL->nods_, FL->mesh_type_);
    const double VOLUME = vc.compute();
    spdlog::info("volume={}", VOLUME);

    //-> give the initial length scale
    if ( FL->cell_.size() != 0 ) { // finest length scale
      const auto &cell = FL->cell_;
      const auto &nods = FL->nods_;
      for (size_t i = 0; i < cell.cols(); ++i) {
        for (size_t j = 0; j < cell.rows(); ++j) {
          for (size_t k = j+1; k < cell.rows(); ++k) {
            const double djk = (nods.col(cell(j, i))-nods.col(cell(k, i))).squaredNorm();
            L[cell(j, i)] = std::max(L[cell(j, i)], djk);
            L[cell(k, i)] = std::max(L[cell(k, i)], djk);
          }
        }
      }
      L = 1.00001*L.cwiseSqrt();
      std::cout << "--- max min finest l: " << L.maxCoeff() << " "
                << L.minCoeff() << std::endl;
    }

    // starting from the finest level, iteratively overwrite    
    for (int i = levels.size()-1; i >= 0; --i) {
      const auto &&L_i = levels[i]->calc_supp_scale(&VOLUME);
      ASSERT(L_i.size() == levels[i]->nods_.cols());
      for (size_t p = 0; p < L_i.size(); ++p) {
        L[p] = std::max(L[p], L_i[p]);
      }
    }

    //-> assign center
    ichol_order_patt_base<val_t>::center_ = ichol_order_patt_base<val_t>::nods_;

    //-> get the largest basis scale
    l_max_ = ichol_order_patt_base<val_t>::l_.maxCoeff();
    l_min_ = ichol_order_patt_base<val_t>::l_.minCoeff();
    spdlog::info("l_max={}, l_min={}", l_max_, l_min_);
  }
  void run(const double rho) {
    //-> build whole fine to coarse ordering
    const size_t N = ichol_order_patt_base<val_t>::center_.cols();

    const auto rd = ichol_order_patt_base<val_t>::Rd_;
    const auto &L = ichol_order_patt_base<val_t>::l_;

    //-> bigger indices are for fine
    ichol_order_patt_base<val_t>::fullP_.resize(rd*N);
    ichol_order_patt_base<val_t>::fullP_.setIdentity();
    ichol_order_patt_base<val_t>::fullP_.indices().reverseInPlace();

    typedef typename ichol_order_patt_base<val_t>::PermutationType PT;   
    const PT &&toMesh = ichol_order_patt_base<val_t>::fullP_.inverse();
    const auto &toLevel = ichol_order_patt_base<val_t>::fullP_;

    const auto &levels = ichol_order_patt_base<val_t>::levels_;

    std::vector<int> level_idx(levels.size()+1);
    level_idx[0] = 0;
    for (size_t i = 0; i < levels.size(); ++i) {
      level_idx[i+1] = levels[i]->nods_.cols();
    }

    spdlog::info("range search start");    
    std::vector<std::vector<size_t>> adj_list(N);
    for (int i = levels.size()-1; i >= 0; --i) { // for each level, start from the finest
      const int start_id = level_idx[i+1];
      const int end_id = level_idx[i];
      spdlog::info("process {} to {}", start_id, end_id);

      //-> construct kd-tree      
      const matd_t &center = levels[i]->nods_;
      EigenCSR pts(center.cols(), center.rows());
      std::copy(center.data(), center.data()+center.size(), pts.data());
      kdt_ = std::make_shared<kd_tree_t>(pts.cols(), pts, 10);
      kdt_->index->buildIndex();
      spdlog::info("curr nodes number={}", center.cols());

      #pragma omp parallel for
      for (int tm_i = start_id-1; tm_i >= end_id; --tm_i) {
        const size_t i = toLevel.indices()[tm_i*rd]/rd;

        //-> do a radius search
        std::vector<std::pair<long int, double>> res;
        const double R = rho*L[tm_i];
        kdt_->index->radiusSearch(&pts(tm_i, 0), R*R, res, nanoflann::SearchParams());

        for (const auto &jd : res) {
          const size_t tm_j = jd.first;
          const size_t j = toLevel.indices()[tm_j*rd]/rd;
          if ( i <= j ) {
            const double dij = (pts.row(tm_i)-pts.row(tm_j)).squaredNorm();
            if ( dij <= rho*rho*std::min(L[tm_i]*L[tm_i], L[tm_j]*L[tm_j]) ) {
              adj_list[i].emplace_back(j);
            }
          }
        }
      }
    }
    #pragma omp parallel for
    for (size_t p = 0; p < N; ++p) {
      std::sort(adj_list[p].begin(), adj_list[p].end());
    }

    const size_t dim = rd*N;
    
    std::vector<ptrdiff_t> ptr, ind;
    std::vector<val_t> val;
    ptr.resize(dim+1);
    size_t nnz = 0;
    ptr[0] = 0;
    for (size_t p = 0; p < N; ++p) {
      size_t nnz_p = rd*adj_list[p].size();
      for (size_t d = 0; d < rd; ++d) {
        ptr[rd*p+d+1] = ptr[rd*p+d]+nnz_p;
        nnz += nnz_p;
      }
    }
    ind.resize(nnz);
    #pragma omp parallel for
    for (size_t i = 0; i < dim; ++i) {
      auto iter = ptr[i];      
      size_t p = i/rd;
      for (size_t q = 0; q < adj_list[p].size(); ++q) {
        for (size_t d = 0; d < rd; ++d) {
          ind[iter++] = rd*adj_list[p][q]+d;
        }
      }
    }
    val.resize(nnz);
    std::fill(val.begin(), val.end(), 0);
    
    ichol_order_patt_base<val_t>::zeroS_ =
        Eigen::Map<typename ichol_order_patt_base<val_t>::PatternType>
        (rd*N, rd*N, nnz, &ptr[0], &ind[0], &val[0]);    
    spdlog::info("range search end with nnz={}", nnz);
  }
    
 private:
  std::shared_ptr<kd_tree_t> kdt_;
  double l_max_, l_min_;
};

template <class SpMat, typename int_type>
Eigen::Matrix<int_type, -1, 1>
group_by_color(const SpMat &A, const int_type begin, const int_type end) {
  const int_type dim = end-begin;
  Eigen::Matrix<int_type, -1, 1> order(dim);
  
  std::vector<int_type> colored(dim, 0);
  std::vector<bool> interact(dim, 0);
  
  int_type e = 1, count = 0;
  while ( count < dim ) {
    std::fill(interact.begin(), interact.end(), 0);
    
    for (int_type j = begin; j < end; ++j) {
      if ( colored[j-begin] == 0 && interact[j-begin] == false ) {
        colored[j-begin] = e;
        order[count++] = j;

        for (typename SpMat::InnerIterator it(A, j); it; ++it) {
          const int_type i = it.row();
          if ( i < begin ) continue;
          if ( i >= end ) break;
          if ( i != j ) interact[i-begin] = true;
        }
      }
    }

    ++e;
  }

  spdlog::info("color num of block: {}", e-1);  
  return order;
}

template <typename T, typename INT>
Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
adjust_inlevel_order(const Eigen::SparseMatrix<T, Eigen::ColMajor, INT> &A,
                     const std::vector<std::ptrdiff_t> &level_ptr,
                     const int option=0) {
  ASSERT(A.rows() == level_ptr.back());

  Eigen::VectorXi new_order(A.rows());

  if ( option == 0 ) {
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < level_ptr.size()-1; ++i) {
      const int begin = level_ptr[i], end = level_ptr[i+1], dim = end-begin;
      new_order.segment(begin, dim) = group_by_color(A, begin, end);
    }
  } else if ( option == 1 ) {
    const int begin = level_ptr.front(), end = level_ptr.back(), dim = end-begin;
    ASSERT(begin == 0 && A.rows() == dim);
    new_order = group_by_color(A, begin, end);
  } else if ( option == 2 ) {
    ASSERT(level_ptr.size() == 3);
    for (int i = 0; i < level_ptr.size()-1; ++i) {
      const int begin = level_ptr[i], end = level_ptr[i+1], dim = end-begin;
      if ( i == 0 ) {
        new_order.segment(begin, dim) = group_by_color(A, begin, end);
      } else {
        for (int p = begin; p < end; ++p) {
          new_order[p] = p;
        }
      }
    }
  } else {
    bool unsupported_option_in_multicoloring = false;
    ASSERT(unsupported_option_in_multicoloring);
  }

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> P(A.rows());
  for (int i = 0; i < new_order.size(); ++i) {
    P.indices()[new_order[i]] = i;
  }

  return P;
}

}

#endif
