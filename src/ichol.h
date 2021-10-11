#ifndef SUPER_COLUMN_H
#define SUPER_COLUMN_H

#include <numeric>
#include <fstream>
#include <Eigen/Sparse>
#include <spdlog/spdlog.h>
#include <cblas.h>
#include <set>
#include <unordered_set>
#include <bitset>

#include <egl/writeDMAT.h>
#include "timer.h"

namespace mschol
{

  typedef std::bitset<64> mask_type;

  extern "C" int dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);

  template <typename INT>
  struct static_graph_t
  {
    std::vector<INT> u_, v_;
    std::vector<INT> in_degree_;
    std::vector<INT> first_, next_;
    std::vector<bool> valid_;
    INT curr_edge_;

    static_graph_t(const INT vert_num, const INT edge_num)
        : curr_edge_(0)
    {
      in_degree_.resize(vert_num);
      std::fill(in_degree_.begin(), in_degree_.end(), 0);

      u_.resize(edge_num);
      v_.resize(edge_num);
      valid_.resize(edge_num);
      first_.resize(vert_num);
      next_.resize(edge_num);
      std::fill(first_.begin(), first_.end(), -1);
      std::fill(next_.begin(), next_.end(), -1);
      std::fill(valid_.begin(), valid_.end(), false);
    }
    INT num_nodes() const
    {
      return first_.size();
    }
    void add_edge(const INT u, const INT v)
    { // u-->v
      u_[curr_edge_] = u;
      v_[curr_edge_] = v;
      valid_[curr_edge_] = true;
      ++in_degree_[v];

      next_[curr_edge_] = first_[u];
      first_[u] = curr_edge_;

      ++curr_edge_;
    }
    INT in_degree(const INT pid)
    {
      return in_degree_[pid];
    }
    void remove_out_edges(const INT pid)
    {
      INT e = first_[pid];
      while (e != -1)
      {
        valid_[e] = false;
        --in_degree_[v_[e]];
        e = next_[e];
      }
    }
  };

  template <class SpMat, typename index_t>
  static void mt_build_level_(const SpMat &A,
                              std::vector<index_t> &lev_ptr,
                              std::vector<index_t> &lev_ind,
                              const bool forward)
  {
    ASSERT(A.isCompressed());

    const index_t dim = A.cols(), nnz = A.nonZeros();
    lev_ptr.clear();
    lev_ind.clear();
    lev_ind.reserve(dim);

    const auto ptr = A.outerIndexPtr();
    const auto ind = A.innerIndexPtr();

    //-> construct the graph
    static_graph_t<index_t> g(dim, nnz - dim);

    if (forward)
    {
      for (index_t j = 0; j < dim; ++j)
      {
        for (index_t cnt = ptr[j]; cnt < ptr[j + 1] - 1; ++cnt)
        {
          const index_t i = ind[cnt];
          g.add_edge(i, j);
        }
      }
    }
    else
    {
      for (index_t j = 0; j < dim; ++j)
      {
        for (index_t cnt = ptr[j] + 1; cnt < ptr[j + 1]; ++cnt)
        {
          const index_t i = ind[cnt];
          g.add_edge(i, j);
        }
      }
    }
    spdlog::info("edges added");

    std::list<index_t> alive_node(dim);
    std::iota(alive_node.begin(), alive_node.end(), 0);

    lev_ptr.push_back(0);
    while (!alive_node.empty())
    {
      index_t root_cnt = 0;

      auto iter = alive_node.begin();
      while (iter != alive_node.end())
      {
        if (g.in_degree(*iter) == 0)
        {
          ++root_cnt;
          lev_ind.push_back(*iter);
          iter = alive_node.erase(iter);
        }
        else
        {
          ++iter;
        }
      }

      index_t offset = lev_ptr.back() + root_cnt;
      lev_ptr.push_back(offset);

      const index_t n = lev_ptr.size();
      for (index_t iter = lev_ptr[n - 2]; iter < lev_ptr[n - 1]; ++iter)
      {
        const index_t vid = lev_ind[iter];
        g.remove_out_edges(vid);
      }
    }
    spdlog::info("level scheduling done");
    ASSERT(lev_ind.size() == dim);
  }

  template <typename T, typename index_t, int ORDER = Eigen::ColMajor>
  struct supcol_sparse_matrix
  {
    std::vector<index_t> sup_ptr_, sup_ind_;
    std::vector<index_t> col2sup_, col2off_;
    Eigen::SparseMatrix<index_t, ORDER> sup_U_;
    Eigen::SparseMatrix<mask_type, ORDER> mask_;
    Eigen::Matrix<T, -1, 1> block_;
  };

  template <typename T, typename index_t, int ORDER = Eigen::ColMajor>
  struct supcol_sparse_matrix_ref
  {
    std::vector<index_t> *sup_ptr_, *sup_ind_;
    std::vector<index_t> *col2sup_, *col2off_;
    Eigen::SparseMatrix<index_t, ORDER> sup_U_;
    Eigen::SparseMatrix<mask_type, ORDER> mask_;
    Eigen::Matrix<T, -1, 1> *block_;
  };

  template <class Container>
  int sum_before(const Container &m, const int p)
  {
    int cnt = 0;
    for (int i = 0; i < p; ++i)
    {
      if (m[i])
      {
        ++cnt;
      }
    }
    return cnt;
  }

  template <typename T, typename index_t>
  struct supcol_comp_kernel
  {
    template <int Order>
    static inline index_t num_supernodes(const supcol_sparse_matrix<T, index_t, Order> &mat)
    {
      return mat.sup_ptr_.size() - 1;
    }
    template <int Order>
    static inline index_t num_supernodes(const supcol_sparse_matrix_ref<T, index_t, Order> &mat)
    {
      return mat.sup_ptr_->size() - 1;
    }
    template <int Order>
    static inline index_t node_size(const supcol_sparse_matrix<T, index_t, Order> &mat, const index_t VID)
    {
      return mat.sup_ptr_[VID + 1] - mat.sup_ptr_[VID];
    }
    template <int Order>
    static inline index_t node_size(const supcol_sparse_matrix_ref<T, index_t, Order> &mat, const index_t VID)
    {
      return (*mat.sup_ptr_)[VID + 1] - (*mat.sup_ptr_)[VID];
    }
    template <int Order>
    static inline index_t head_index(const supcol_sparse_matrix<T, index_t, Order> &mat, const index_t VID)
    {
      return mat.sup_ind_[mat.sup_ptr_[VID]];
    }
    template <int Order>
    static inline index_t head_index(const supcol_sparse_matrix_ref<T, index_t, Order> &mat, const index_t VID)
    {
      return (*mat.sup_ind_)[(*mat.sup_ptr_)[VID]];
    }
    // sn_U [upper triangular matrix] records the connectivity between supernodes
    static void init(const Eigen::SparseMatrix<index_t> &sn_U,
                     const Eigen::SparseMatrix<mask_type> &mask,
                     const std::vector<index_t> &sn_ptr,
                     const std::vector<index_t> &sn_ind,
                     supcol_sparse_matrix<T, index_t> &mat)
    {
      const index_t dim = sn_ind.size();         // size of all DoFs
      const index_t num_spn = sn_ptr.size() - 1; // size of supernodes
      ASSERT(num_spn == sn_U.rows());

      mat.sup_ptr_ = sn_ptr;
      mat.sup_ind_ = sn_ind;

      mat.col2sup_.resize(dim);
      mat.col2off_.resize(dim);
      for (index_t i = 0; i < num_spn; ++i)
      {
        for (index_t cnt = sn_ptr[i]; cnt < sn_ptr[i + 1]; ++cnt)
        {
          const index_t pid = sn_ind[cnt];
          mat.col2sup_[pid] = i;
          mat.col2off_[pid] = cnt - sn_ptr[i];
        }
      }

      mat.sup_U_ = sn_U;
      if (!mat.sup_U_.isCompressed())
      {
        mat.sup_U_.makeCompressed();
      }
      mat.mask_ = mask;
      if (!mat.mask_.isCompressed())
      {
        mat.mask_.makeCompressed();
      }

      std::vector<std::vector<std::pair<index_t, index_t>>> ind_itr(num_spn);
      index_t fillin_num = 0;
      for (index_t J = 0; J < mat.sup_U_.cols(); ++J)
      {
        for (index_t iter = mat.sup_U_.outerIndexPtr()[J]; iter < mat.sup_U_.outerIndexPtr()[J + 1]; ++iter)
        {
          const index_t I = mat.sup_U_.innerIndexPtr()[iter];
          fillin_num += mat.mask_.valuePtr()[iter].count() * (mat.sup_ptr_[I + 1] - mat.sup_ptr_[I]);
          ind_itr[I].emplace_back(std::make_pair(J, iter));
        }
      }
      mat.block_.setZero(fillin_num);
      spdlog::info("nnz in superU: {}", mat.sup_U_.nonZeros());
      spdlog::info("nnz in supernodal block: {}", fillin_num);
      spdlog::info("total size: {}", 1.0 * dim * dim);

      index_t count = 0;
      for (index_t I = 0; I < ind_itr.size(); ++I)
      {
        for (index_t k = 0; k < ind_itr[I].size(); ++k)
        {
          index_t iter = ind_itr[I][k].second;
          mat.sup_U_.valuePtr()[iter] = count;
          count += mat.mask_.valuePtr()[iter].count() * (mat.sup_ptr_[I + 1] - mat.sup_ptr_[I]);
        }
      }
      ASSERT(fillin_num == count);
    }
    static void scale_sup_diag(supcol_sparse_matrix<T, index_t, Eigen::ColMajor> &mat,
                               const index_t J, const T s)
    {
      const auto SU_ptr = mat.sup_U_.outerIndexPtr();
      const auto SU_ind = mat.sup_U_.innerIndexPtr();
      const auto SU_val = mat.sup_U_.valuePtr();

      const index_t numJ = node_size(mat, J);

      T *v = &mat.block_[SU_val[SU_ptr[J + 1] - 1]];
      Eigen::Map<Eigen::Matrix<T, -1, -1>> Ujj(v, numJ, numJ);
      for (index_t i = 0; i < numJ; ++i)
      {
        Ujj(i, i) *= s;
      }
    }
    static void set_col_zero(supcol_sparse_matrix<T, index_t, Eigen::ColMajor> &mat,
                             const index_t J)
    {
      const auto SU_ptr = mat.sup_U_.outerIndexPtr();
      const auto SU_ind = mat.sup_U_.innerIndexPtr();
      const auto SU_val = mat.sup_U_.valuePtr();
      const auto MA_val = mat.mask_.valuePtr();

      for (index_t iter = SU_ptr[J]; iter < SU_ptr[J + 1]; ++iter)
      {
        const index_t I = SU_ind[iter];
        const index_t numI = node_size(mat, I);
        const index_t numJ = MA_val[iter].count();
        T *v = &mat.block_[SU_val[iter]];
        std::fill(v, v + numI * numJ, 0);
      }
    }
    static void conv_csc_to_csr(const supcol_sparse_matrix<T, index_t, Eigen::ColMajor> &A,
                                supcol_sparse_matrix_ref<T, index_t, Eigen::RowMajor> &B)
    {
      B.sup_ptr_ = const_cast<std::vector<index_t> *>(&A.sup_ptr_);
      B.sup_ind_ = const_cast<std::vector<index_t> *>(&A.sup_ind_);
      B.col2sup_ = const_cast<std::vector<index_t> *>(&A.col2sup_);
      B.col2off_ = const_cast<std::vector<index_t> *>(&A.col2off_);
      B.sup_U_ = A.sup_U_;
      B.mask_ = A.mask_;
      B.block_ = const_cast<Eigen::Matrix<T, -1, 1> *>(&A.block_);
    }
    template <int ORDER>
    static void transpose(const supcol_sparse_matrix_ref<T, index_t, ORDER> &A,
                          supcol_sparse_matrix_ref<T, index_t, ORDER> &AT)
    {
      AT.sup_ptr_ = A.sup_ptr_;
      AT.sup_ind_ = A.sup_ind_;
      AT.col2sup_ = A.col2sup_;
      AT.col2off_ = A.col2off_;
      AT.sup_U_ = A.sup_U_.transpose();
      AT.mask_ = A.mask_.transpose();
      AT.block_ = A.block_;
    }
  };

  //// a supernodal multi-threaded IC solver
  template <typename T, typename index_t>
  class supcol_ichol_solver
  {
  public:
    typedef Eigen::Matrix<T, -1, 1> vec_type;

    supcol_ichol_solver(const T ALPHA) : ALPHA_(ALPHA)
    {
      spdlog::info("INITIAL ALPHA={}", ALPHA);
    }
    void symbolic_phase(const Eigen::SparseMatrix<T> &A,
                        const Eigen::SparseMatrix<index_t> &supU,
                        const Eigen::SparseMatrix<mask_type> &mask,
                        const std::vector<index_t> &sn_ptr,
                        const std::vector<index_t> &sn_ind)
    {
      ASSERT(sn_ptr.size() == supU.rows() + 1 && A.rows() == sn_ind.size());
      ASSERT(A.isCompressed());

      //-> A is a sparse SPD matrix and get its upper triangular part
      supcol_comp_kernel<T, index_t>::init(supU, mask, sn_ptr, sn_ind, SU_);

//-> level scheduling
#pragma omp parallel sections
      {
#pragma omp section
        {
          mt_build_level_(SU_.sup_U_, sn_lev_ptr_, sn_lev_ind_, true);
        }

#pragma omp section
        {
          decltype(SU_.sup_U_) UT = std::move(SU_.sup_U_.transpose());
          mt_build_level_(UT, bw_lev_ptr_, bw_lev_ind_, false);
        }
      }

      nnz_pos_.resize(A.nonZeros());
      std::fill(nnz_pos_.begin(), nnz_pos_.end(), -1);
#pragma omp parallel for
      for (index_t j = 0; j < A.cols(); ++j)
      {
        index_t J = SU_.col2sup_[j];

        index_t iter = A.outerIndexPtr()[j], ITER = SU_.sup_U_.outerIndexPtr()[J];
        while (iter < A.outerIndexPtr()[j + 1] && ITER < SU_.sup_U_.outerIndexPtr()[J + 1])
        {
          index_t i = A.innerIndexPtr()[iter];
          index_t I = SU_.sup_U_.innerIndexPtr()[ITER];
          if (SU_.col2sup_[i] == I)
          {
            index_t numI = supcol_comp_kernel<T, index_t>::node_size(SU_, I);
            index_t off_i = SU_.col2off_[i], off_j = SU_.col2off_[j];
            index_t real_off_j = sum_before(SU_.mask_.valuePtr()[ITER], off_j);
            nnz_pos_[iter] = SU_.sup_U_.valuePtr()[ITER] + off_i + real_off_j * numI;
            ++iter;
          }
          else if (SU_.col2sup_[i] < I)
          {
            ++iter;
          }
          else
          {
            ++ITER;
          }
        }
      }

      const double total_size = (double)A.rows() * (double)A.cols();
      spdlog::info("factor density: {}", SU_.block_.size() / total_size);
    }
    int factorize(const Eigen::SparseMatrix<T> &A, const int num_threads = 8)
    {
      ASSERT(A.isCompressed());
      T alpha = ALPHA_;

      spdlog::info("init ichol...");
      auto A_val = A.valuePtr();
      auto A_ptr = A.outerIndexPtr();
      auto A_ind = A.innerIndexPtr();
      {
        SU_.block_.setZero();
#pragma omp parallel for
        for (index_t i = 0; i < A.nonZeros(); ++i)
        {
          if (nnz_pos_[i] >= 0)
          {
            SU_.block_[nnz_pos_[i]] = A_val[i];
          }
        }
      }

      const index_t lev_num = sn_lev_ptr_.size() - 1;
      const index_t num_su_nodes = SU_.sup_U_.rows();
      spdlog::info("supernode number={}", num_su_nodes);

      //-> record the history of diagonal shiftings
      Eigen::VectorXi vis = Eigen::VectorXi::Zero(num_su_nodes);
      std::vector<bool> factorized(num_su_nodes, false);

      bool ic_success;
      index_t num_breakdown = 0;
      do
      {
        ic_success = true;

        for (index_t p = 0; p < lev_num; ++p)
        {
          const index_t lev_begin = sn_lev_ptr_[p];
          const index_t lev_end = sn_lev_ptr_[p + 1];
          const index_t count = lev_end - lev_begin;

          Eigen::VectorXi flag(count);
          flag.setZero();

          int nb_threads = std::min(num_threads, static_cast<int>(count));
#pragma omp parallel for num_threads(nb_threads)
          for (index_t cnt = lev_begin; cnt < lev_end; ++cnt)
          {
            if (!factorized[sn_lev_ind_[cnt]])
            {
              flag[cnt - lev_begin] = iteration_body(sn_lev_ind_[cnt]);
              if (flag[cnt - lev_begin] == 0)
              {
                factorized[sn_lev_ind_[cnt]] = true;
              }
            }
          }
          if (flag.sum() > 0)
          { // for rebooting
            ic_success = false;
            ++num_breakdown;

            const auto PTR = SU_.sup_U_.outerIndexPtr();
            const auto IND = SU_.sup_U_.innerIndexPtr();

            //-> for each negative supernodal pivot
            std::set<index_t> redo_J;
            for (index_t i = 0; i < flag.size(); ++i)
            {
              if (flag[i] == 0)
                continue;

              const index_t J = sn_lev_ind_[i + lev_begin];
              std::cout << "breakdonws at supernode " << J << std::endl;
              std::cout << "affected by " << PTR[J + 1] - PTR[J] << " cols" << std::endl;

              for (index_t iter = PTR[J]; iter < PTR[J + 1]; ++iter)
              {
                const index_t K = IND[iter];
                ++vis[K];
                redo_J.insert(K);
                factorized[K] = false;
              }
            }

            for (index_t J = 0; J < num_su_nodes; ++J)
            {
              if (factorized[J])
              {
                //-> for already factorized cols
                for (index_t iter = PTR[J]; iter < PTR[J + 1]; ++iter)
                {
                  const index_t I = IND[iter];
                  if (redo_J.find(I) != redo_J.end())
                  {
                    factorized[J] = false;
                    break;
                  }
                }
              }
            }

// for all unfactorized cols, re-assign their values
#pragma omp parallel for
            for (index_t J = 0; J < num_su_nodes; ++J)
            {
              if (!factorized[J])
              {
                supcol_comp_kernel<T, index_t>::set_col_zero(SU_, J);

                index_t head = supcol_comp_kernel<T, index_t>::head_index(SU_, J);
                index_t numJ = supcol_comp_kernel<T, index_t>::node_size(SU_, J);
                for (index_t col = head; col < head + numJ; ++col)
                {
                  for (index_t iter = A_ptr[col]; iter < A_ptr[col + 1]; ++iter)
                  {
                    if (nnz_pos_[iter] >= 0)
                    {
                      SU_.block_[nnz_pos_[iter]] = A_val[iter];
                    }
                  }
                }
                if (vis[J] > 0)
                {
                  const T s = 1 + std::pow(2, vis[J] - 1) * alpha;
                  supcol_comp_kernel<T, index_t>::scale_sup_diag(SU_, J, s);
                }
              }
            }

            break;
          }
        }
      } while (!ic_success);

      supcol_comp_kernel<T, index_t>::conv_csc_to_csr(SU_, SLT_);
      supcol_comp_kernel<T, index_t>::transpose(SLT_, SL_);
      spdlog::info("reboot times: {}", num_breakdown);

      return 0;
    }
    template <class Vec>
    void solve(const Vec &b, Vec &x, const int num_threads = 8)
    {
      x = b;

      //-> forward
      for (index_t p = 0; p < sn_lev_ptr_.size() - 1; ++p)
      {
        const index_t lev_begin = sn_lev_ptr_[p];
        const index_t lev_end = sn_lev_ptr_[p + 1];

        int nb_threads = std::min(num_threads, static_cast<int>(lev_end - lev_begin));
#pragma omp parallel for num_threads(nb_threads)
        for (index_t cnt = lev_begin; cnt < lev_end; ++cnt)
        {
          fw_subst_body(sn_lev_ind_[cnt], x);
        }
      }

      //-> backward
      for (index_t p = 0; p < bw_lev_ptr_.size() - 1; ++p)
      {
        const index_t lev_begin = bw_lev_ptr_[p];
        const index_t lev_end = bw_lev_ptr_[p + 1];

        int nb_threads = std::min(num_threads, static_cast<int>(lev_end - lev_begin));
#pragma omp parallel for num_threads(nb_threads)
        for (index_t cnt = lev_begin; cnt < lev_end; ++cnt)
        {
          bw_subst_body(bw_lev_ind_[cnt], x);
        }
      }
    }

  private:
    int iteration_body(const index_t J)
    {
      // spdlog::info("iteration body {}", J);
      const auto SU_ptr = SU_.sup_U_.outerIndexPtr();
      const auto SU_ind = SU_.sup_U_.innerIndexPtr();
      const auto SU_val = SU_.sup_U_.valuePtr();
      const auto MA_val = SU_.mask_.valuePtr();

      //-> j now is the index of supernode
      const index_t begin_J = SU_ptr[J];
      const index_t end_J = SU_ptr[J + 1] - 1;
      int numJ = supcol_comp_kernel<T, index_t>::node_size(SU_, J);

      for (index_t cnt = SU_ptr[J]; cnt < SU_ptr[J + 1]; ++cnt)
      {
        const index_t I = SU_ind[cnt];
        const index_t begin_I = SU_ptr[I];
        const index_t end_I = SU_ptr[I + 1] - 1;
        int numI = supcol_comp_kernel<T, index_t>::node_size(SU_, I);
        T *S = &SU_.block_[SU_val[cnt]];

        index_t iter1 = begin_I, iter2 = begin_J;
        while (iter1 < end_I && iter2 < end_J)
        {
          if (SU_ind[iter1] == SU_ind[iter2])
          {
            index_t K = SU_ind[iter1];
            int numK = supcol_comp_kernel<T, index_t>::node_size(SU_, K);

            auto p = MA_val[iter1].count(), q = MA_val[iter2].count();
            Eigen::Matrix<T, -1, -1> s(p, q);
            s.setZero();
            T *a = &SU_.block_[SU_val[iter1]];
            T *b = &SU_.block_[SU_val[iter2]];
            // spdlog::info("U{}{}-=U{}{} * U{}{}, numK={}, iter1={},iter2={} p={},q={}",
            //              I, J, I, K, J, K, numK, iter1, iter2, p, q);
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, p, q, numK, 1,
                        a, numK, b, numK, 1, s.data(), p);

            // scatter S
            std::vector<index_t> rows(p);
            int count = 0;
            for (int k = 0; k < MA_val[iter1].size() && count < p; ++k)
            {
              if (MA_val[iter1][k])
              {
                rows[count++] = k;
              }
            }
            const auto &intersect = MA_val[iter2] & MA_val[cnt];
            index_t offB = 0, offC = 0;
            for (int k = 0; k < intersect.size() && offB < q; ++k)
            {
              if (intersect[k])
              {
                for (int m = 0; m < p; ++m)
                {
                  *(S + rows[m] + numI * offC) -= s(m, offB);
                }
              }
              if (MA_val[iter2][k])
              {
                ++offB;
              }
              if (MA_val[cnt][k])
              {
                ++offC;
              }
            }

            ++iter1;
            ++iter2;
          }
          else if (SU_ind[iter1] < SU_ind[iter2])
          {
            ++iter1;
          }
          else
          {
            ++iter2;
          }
        }

        if (I < J)
        {
          //-> lower triangular solve
          T *Uii = &SU_.block_[SU_val[SU_ptr[I + 1] - 1]];
          T *Uij = &SU_.block_[SU_val[cnt]];
          int n = SU_.mask_.valuePtr()[cnt].count();
          // spdlog::info("Uij/Uii: I,J={},{} numI={},n={}", I, J, numI, n);
          cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, numI, n, 1, Uii, numI, Uij, numI);
        }
        else
        {
          //-> dense cholesky
          // spdlog::info("dense cholesky numJ={}", numJ);
          T *Ujj = &SU_.block_[SU_val[cnt]];
          matd_t mat = Eigen::Map<matd_t>(Ujj, numJ, numJ);
          // egl::writeDMAT("Ujj.mat", mat, false);
          char uplo = 'U';
          int info;
          dpotrf_(&uplo, &numJ, Ujj, &numJ, &info);
          // spdlog::info("dense cholesky info={}", info);
          if (info != 0)
          {
            return 1;
          }
        }
      }

      return 0;
    }
    template <class Vec>
    void fw_subst_body(const index_t I, Vec &x)
    {
      const index_t begin = SL_.sup_U_.outerIndexPtr()[I];
      const index_t end = SL_.sup_U_.outerIndexPtr()[I + 1] - 1;
      const auto IND = SL_.sup_U_.innerIndexPtr();
      const auto VAL = SL_.sup_U_.valuePtr();
      const auto MA_VAL = SL_.mask_.valuePtr();

      const index_t head_I = supcol_comp_kernel<T, index_t>::head_index(SL_, I);
      const index_t numI = supcol_comp_kernel<T, index_t>::node_size(SL_, I);

      for (index_t iter = begin; iter < end; ++iter)
      {
        const index_t J = IND[iter];
        const int numJ = supcol_comp_kernel<T, index_t>::node_size(SL_, J);

        const index_t m = MA_VAL[iter].count();
        Vec res(m);
        res.setZero();

        const T *A = &(*SL_.block_)[VAL[iter]];
        const index_t head_J = supcol_comp_kernel<T, index_t>::head_index(SL_, J);
        const T *v = &x[head_J];
        cblas_dgemv(CblasColMajor, CblasTrans, numJ, m, 1, A, numJ, v, 1, 1, res.data(), 1);

        // dispatch
        int cnt = 0;
        const auto &MA = MA_VAL[iter];
        for (int k = 0; k < MA.size() && cnt < m; ++k)
        {
          if (MA[k])
          {
            x[head_I + k] -= res[cnt++];
          }
        }
      }

      const T *Lii = &(*SL_.block_)[VAL[end]];
      cblas_dtrsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit, numI, Lii, numI, &x[head_I], 1);
    }
    template <class Vec>
    void bw_subst_body(const index_t I, Vec &x)
    {
      const index_t begin = SLT_.sup_U_.outerIndexPtr()[I] + 1;
      const index_t end = SLT_.sup_U_.outerIndexPtr()[I + 1];
      const auto IND = SLT_.sup_U_.innerIndexPtr();
      const auto VAL = SLT_.sup_U_.valuePtr();
      const auto MA_VAL = SLT_.mask_.valuePtr();

      const index_t head_I = supcol_comp_kernel<T, index_t>::head_index(SLT_, I);
      const index_t numI = supcol_comp_kernel<T, index_t>::node_size(SLT_, I);

      int M = (VAL[end] - VAL[begin]) / numI;
      Vec v(M);
      v.setZero();

      // collect
      index_t count = 0;
      for (index_t iter = begin; iter < end; ++iter)
      {
        const index_t J = IND[iter];
        const index_t head_J = supcol_comp_kernel<T, index_t>::head_index(SLT_, J);
        const auto &MA = MA_VAL[iter];
        for (int k = 0; k < MA.size(); ++k)
        {
          if (MA[k])
          {
            v[count++] = x[head_J + k];
          }
        }
      }

      const T *A = &(*SLT_.block_)[VAL[begin]];
      cblas_dgemv(CblasColMajor, CblasNoTrans, numI, M, -1, A, numI, v.data(), 1, 1, &x[head_I], 1);

      const T *Uii = &(*SLT_.block_)[VAL[begin - 1]];
      cblas_dtrsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit, numI, Uii, numI, &x[head_I], 1);
    }

  public:
    const T ALPHA_;

    std::vector<index_t> nnz_pos_;

    supcol_sparse_matrix<T, index_t, Eigen::ColMajor> SU_;
    supcol_sparse_matrix_ref<T, index_t, Eigen::RowMajor> SL_, SLT_;

    std::vector<index_t> sn_lev_ptr_, sn_lev_ind_; // forward scheduling
    std::vector<index_t> bw_lev_ptr_, bw_lev_ind_; // backward scheduling
  };

}

#endif
