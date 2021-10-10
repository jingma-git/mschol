#include <iostream>
#include <spdlog/spdlog.h>

#include "io.h"
#include "macro.h"
#include "ichol_pattern.h"
#include "supernode.h"
#include "cholesky.h"
#include "ichol.h"
// #include "visual.h"

using namespace std;
using namespace Eigen;

namespace mschol
{

  ichol_precond::ichol_precond(const vector<shared_ptr<chol_level>> &levels,
                               const boost::property_tree::ptree &pt)
      : levels_(levels), pt_(pt)
  {
    level_ptr_.reserve(levels.size() + 1);

    std::ptrdiff_t offset = 0;
    for (std::ptrdiff_t i = levels.size() - 1; i >= 0; --i)
    {
      level_ptr_.push_back(offset);
      offset += levels[i]->W_.rows();
    }
    level_ptr_.push_back(levels.back()->W_.cols());

    cout << "--- level ptr: ";
    print_vector(level_ptr_);
    cout << endl;

    const double nei_num = pt_.get<double>("nei_num.value", 1.001);
    const std::ptrdiff_t max_su_size = pt_.get<std::ptrdiff_t>("max_su_size.value");
    spdlog::info("nei_num {}", nei_num);
    spdlog::info("max supernode size: {}", max_su_size);

    high_resolution_timer hrt;
    hrt.start();
    spdlog::info("====== build pattern and wavelet trans ======");
    std::shared_ptr<ichol_order_patt_base<char>> order_patt;
    order_patt = std::make_shared<ichol_ftoc_order<char>>(levels_, 'D');

    order_patt->run(nei_num);
    hrt.stop();
    spdlog::info("run sparsity time={}", hrt.duration() / 1000);
    const std::string outdir = pt_.get<std::string>("outdir.value");
    // vis_spmat(outdir + "/S0.png", order_patt->getZeroS());

    //-> fine to coarse ordering
    P_ = order_patt->getFullP();

    spdlog::info("====== aggregate supernodes ======");
    gs_ = std::make_shared<geom_supernode>();
    const auto &&agP = gs_->aggregate(levels_, order_patt->getL(),
                                      order_patt->getFullP(),
                                      order_patt->getZeroS(), nei_num / 2.0,
                                      max_su_size);

    //-> aggregation ordering
    P_ = agP * P_;
    PermutationMatrix<-1, -1> P = agP;

    const auto &&mcP = gs_->multicoloring();

    //-> multicolor ordering
    P = mcP * P;
    P_ = mcP * P_;

    typedef geom_supernode::index_t index_t;
    typedef geom_supernode::PattType patt_t;

    const index_t dim = P.rows();
    const index_t SU_NUM = gs_->S_su_.rows();

    std::vector<index_t> col_to_su(dim), col_to_of(dim);
    for (int i = 0; i < gs_->sn_ptr_.size() - 1; ++i)
    {
      for (int cnt = gs_->sn_ptr_[i]; cnt < gs_->sn_ptr_[i + 1]; ++cnt)
      {
        const int col_idx = gs_->sn_ind_[cnt];
        col_to_su[col_idx] = i;
        col_to_of[col_idx] = cnt - gs_->sn_ptr_[i];
      }
    }

    spdlog::info("compute supernodal cell row mask......");
    std::unordered_map<index_t, index_t> IJ_to_iter;
    for (index_t J = 0; J < gs_->S_su_.cols(); ++J)
    {
      for (index_t iter = gs_->S_su_.outerIndexPtr()[J]; iter < gs_->S_su_.outerIndexPtr()[J + 1]; ++iter)
      {
        const index_t I = gs_->S_su_.innerIndexPtr()[iter];
        IJ_to_iter.insert(std::make_pair(I * SU_NUM + J, iter));
      }
    }

    //-> lower triangular matrix
    spdlog::info("initialize masks");
    const auto &S = order_patt->getZeroS();
    std::vector<mask_type> tmp_mask(gs_->S_su_.nonZeros());
    for (index_t j = 0; j < S.cols(); ++j)
    {
      for (geom_supernode::PattType::InnerIterator it(S, j); it; ++it)
      {
        index_t I = col_to_su[P.indices()[it.row()]];
        index_t J = col_to_su[P.indices()[it.col()]];
        if (I == J)
        {
          auto it = IJ_to_iter.find(I * SU_NUM + J);
          ASSERT(it != IJ_to_iter.end());
          if (tmp_mask[it->second].none())
          {
            for (index_t k = 0; k < gs_->sn_ptr_[I + 1] - gs_->sn_ptr_[I]; ++k)
            {
              tmp_mask[it->second][k] = 1;
            }
          }
        }
        else if (I < J)
        {
          const index_t off_J = col_to_of[P.indices()[it.col()]];
          auto it = IJ_to_iter.find(I * SU_NUM + J);
          ASSERT(it != IJ_to_iter.end());
          tmp_mask[it->second][off_J] = 1;
        }

        I = col_to_su[P.indices()[it.col()]];
        J = col_to_su[P.indices()[it.row()]];
        if (I == J)
        {
          auto it = IJ_to_iter.find(I * SU_NUM + J);
          ASSERT(it != IJ_to_iter.end());
          if (tmp_mask[it->second].none() == 0)
          {
            for (index_t k = 0; k < gs_->sn_ptr_[I + 1] - gs_->sn_ptr_[I]; ++k)
            {
              tmp_mask[it->second][k] = 1;
            }
          }
        }
        else if (I < J)
        {
          const index_t off_J = col_to_of[P.indices()[it.row()]];
          auto it = IJ_to_iter.find(I * SU_NUM + J);
          ASSERT(it != IJ_to_iter.end());
          tmp_mask[it->second][off_J] = 1;
        }
      }
    }
    gs_->S_MA_ = Eigen::Map<Eigen::SparseMatrix<mask_type>>(gs_->S_su_.rows(), gs_->S_su_.cols(),
                                                            gs_->S_su_.nonZeros(),
                                                            gs_->S_su_.outerIndexPtr(),
                                                            gs_->S_su_.innerIndexPtr(),
                                                            &tmp_mask[0]);

    const double ALPHA = pt.get<double>("alpha.value");
    ic_slv_ = make_shared<supcol_ichol_solver<double, std::ptrdiff_t>>(ALPHA);
    spdlog::info("====== precomputation done ======");
  }

  int ichol_precond::analyse_pattern(const MatrixType &mat)
  {
    ASSERT(mat.rows() == level_ptr_.back());

    spdlog::info("mat size: {}x{}", mat.rows(), mat.cols());
    spdlog::info("mat nnz: {}", mat.nonZeros());
    spdlog::info("====== symbolic ======");
    SparseMatrix<double> symbA;
    symbA = mat.twistedBy(P_);
    ic_slv_->symbolic_phase(symbA, gs_->S_su_, gs_->S_MA_, gs_->sn_ptr_, gs_->sn_ind_);

    return 0;
  }

  int ichol_precond::factorize(const MatrixType &mat, const bool verbose)
  {
    const int num_threads = pt_.get<int>("num_threads.value");
    spdlog::info("number of threads: {}", num_threads);

    if (!ic_slv_.get())
      return 1;

    G_ = mat.diagonal();
    G_ = G_.cwiseSqrt().cwiseInverse();

    spdlog::info("compuate TAT^t...");
    MatrixType mat_cp = mat;
    const auto PTR = mat_cp.outerIndexPtr();
    const auto IND = mat_cp.innerIndexPtr();
    const auto VAL = mat_cp.valuePtr();
#pragma omp parallel for
    for (size_t j = 0; j < mat_cp.cols(); ++j)
    {
      for (size_t iter = PTR[j]; iter < PTR[j + 1]; ++iter)
      {
        size_t i = IND[iter];
        VAL[iter] *= G_[j] * G_[i];
      }
    }
    MatrixType tmpA;
    tmpA = mat_cp.twistedBy(P_);

    return ic_slv_->factorize(tmpA, num_threads);
  }

  VectorType ichol_precond::solve(const VectorType &rhs)
  {
    static const int subst_num_threads = pt_.get<int>("subst_num_threads.value");

    VectorXd u = P_ * (G_.asDiagonal() * rhs);
    ic_slv_->solve(u, u, subst_num_threads);
    u = G_.asDiagonal() * (P_.inverse() * u);

    return u;
  }

}
