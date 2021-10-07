#ifndef PRECOND_CONJUGATE_GRADIENT_H
#define PRECOND_CONJUGATE_GRADIENT_H

#include <fstream>
#include <Eigen/Sparse>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggr_emin.hpp>
#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include "timer.h"

namespace mschol {

typedef Eigen::SparseMatrix<double> MatrixType;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> CSRMatrixType;
typedef Eigen::VectorXd VectorType;

class preconditioner
{
 public:
  virtual ~preconditioner() {}
  virtual int analyse_pattern(const MatrixType &mat) = 0;
  virtual int factorize(const MatrixType &mat, const bool verbose=true) = 0;
  int compute(const MatrixType &mat, const bool verbose=true) {
    int rtn = 0;
    rtn |= analyse_pattern(mat);
    rtn |= factorize(mat, verbose);
    return rtn;
  }
  virtual VectorType solve(const VectorType &rhs) = 0;
};

class amg_precon : public preconditioner
{
 public:
  typedef amgcl::amg<amgcl::backend::builtin<double>, amgcl::coarsening::ruge_stuben, amgcl::relaxation::spai0> AMG_t;  
  typedef MatrixType::StorageIndex index_t;

  amg_precon(const size_t nrelax, const size_t maxits);
  int analyse_pattern(const MatrixType &mat) { return 0; }
  int factorize(const MatrixType &mat, const bool verbose=true);
  VectorType solve(const VectorType &rhs);
 private:
  const size_t maxits_;
  
  std::vector<index_t> ptr_, ind_;
  std::vector<double> val_;
  std::vector<double> u_;
  
  std::shared_ptr<AMG_t> slv_;
  AMG_t::params prm_;
};

class precond_cg_solver
{
 public:
  typedef Eigen::Index Index;
  typedef double RealScalar;
  typedef double Scalar;

  //-> final error and real iteration number
  RealScalar m_tol_error;
  Index      m_iters;

  precond_cg_solver() { //}: precond_(std::make_shared<identity_precon>()) {
    tol_ = 1e-12;
  }
  precond_cg_solver(const std::shared_ptr<preconditioner> &precond) : precond_(precond) {
    tol_ = 1e-12;
  }

  int analyse_pattern(const MatrixType &mat) {
    return precond_->analyse_pattern(mat);
  }
  int factorize(const MatrixType &mat, const bool verbose=true) {
    mat_    = &mat;
    maxits_ = 2*mat.cols();
    err_.reserve(maxits_);
    time_.reserve(maxits_);    
    return precond_->factorize(mat, verbose);
  }
  
  void set_maxits(const Index maxits) {
    maxits_ = maxits;
    err_.reserve(maxits_);
    time_.reserve(maxits_);
  }
  void set_tol(const RealScalar tol) {
    tol_ = tol;
  }
  VectorType solve(const VectorType &rhs) {
    if ( !time_.empty() ) time_.clear();
    if ( !err_.empty()  ) err_.clear();
    
    Eigen::Map<const CSRMatrixType> MAT(mat_->rows(), mat_->cols(), mat_->nonZeros(),
                                        mat_->outerIndexPtr(), mat_->innerIndexPtr(),
                                        mat_->valuePtr());

    high_resolution_timer timer;
    timer.start();
    
    Index n = mat_->cols();
    VectorType x(n);
    x.setZero();
 
    VectorType residual = rhs-MAT*x; //initial residual 
    RealScalar rhsNorm2 = rhs.squaredNorm();

    time_.emplace_back(timer.duration()/1000.0);
    err_.emplace_back(sqrt(residual.squaredNorm() / rhsNorm2));
    
    if(rhsNorm2 == 0) 
    {
      x.setZero();
      m_iters = 0;
      m_tol_error = 0;
      return x;
    }
    const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();
    RealScalar threshold = std::max(tol_*tol_*rhsNorm2,considerAsZero);
    RealScalar residualNorm2 = residual.squaredNorm();
    if (residualNorm2 < threshold)
    {
      m_iters = 0;
      m_tol_error = sqrt(residualNorm2 / rhsNorm2);
      return x;
    }
 
    VectorType p = precond_->solve(residual);   // initial search direction
 
    VectorType tmp(n);
    RealScalar absNew = residual.dot(p);  // the square of the absolute value of r scaled by invM
    Index i = 0;
    
    while(i < maxits_)
    {
      tmp.noalias() = MAT*p;                  // the bottleneck of the algorithm
 
      Scalar alpha = absNew / p.dot(tmp);         // the amount we travel on dir
      x += alpha * p;                             // update solution
      residual -= alpha * tmp;                    // update residual
     
      residualNorm2 = residual.squaredNorm();
      err_.emplace_back(sqrt(residualNorm2 / rhsNorm2));
      time_.emplace_back(timer.duration()/1000.0);
      
      if(residualNorm2 < threshold)
        break;
     
      const VectorType &&z = precond_->solve(residual); // approximately solve for "A z = residual"
      if ( std::isnan(z.sum()) ) {
        std::cerr << "# preconditioner produced NaN!" << std::endl;
        break;
      }
 
      RealScalar absOld = absNew;
      absNew = residual.dot(z);                   // update the absolute value of r
      RealScalar beta = absNew / absOld;          // calculate the Gram-Schmidt value used to create the new search direction
      p = z + beta * p;                           // update search direction
      i++;
    }
    m_tol_error = sqrt(residualNorm2 / rhsNorm2);
    m_iters = i;

    timer.stop();

    return x;
  }
  int write_profile(const char *outf, const std::string &name) const {
    std::ofstream ofs(outf);
    if ( ofs.fail() ) {
      std::cerr << "# cannot open " << outf << std::endl;
      return __LINE__;
    }
    ofs << "cnt, " << name << std::endl;
    size_t cnt = 0;
    for (const auto &E : err_) {
      ofs << cnt++ << "," << E << std::endl;
    }
    ofs.close();
    return 0;
  }
  int write_timing_profile(const char *outf, const std::string &name) const {
    std::ofstream ofs(outf);
    if ( ofs.fail() ) {
      std::cerr << "# cannot open " << outf << std::endl;
      return __LINE__;
    }
    ofs << "time, " << name << std::endl;
    for (size_t i = 0; i < time_.size(); ++i) {
      ofs << time_[i] << "," << err_[i] << std::endl;
    }
    ofs.close();
    return 0;
  }

 protected:
  MatrixType const *mat_;
  const std::shared_ptr<preconditioner> precond_;

  std::vector<RealScalar> err_, time_;

  Index maxits_;
  RealScalar tol_;
};

}

#endif
