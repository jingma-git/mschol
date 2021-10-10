#include <fstream>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/draw_triangulation_2.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <spdlog/spdlog.h>

#include "src/io.h"
#include "src/chol_hierarchy.h"
#include "src/volume.h"
#include "src/cholesky_level.h"
#include "src/cholesky.h"
#include "src/ptree.h"
#include "src/util.h"
#include "src/visual.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay;
typedef K::Point_2 Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Face_handle Face_handle;

using namespace std;
using namespace mschol;
using namespace Eigen;

static void extr_boundary_node(const mati_t &cell, const matd_t &nods, std::vector<size_t> &bnd)
{
  const double EPS = 1e-10;
  for (size_t i = 0; i < nods.cols(); ++i)
  {
    if (fabs(nods(0, i) - 0) < EPS || fabs(nods(1, i) - 0) < EPS ||
        fabs(nods(0, i) - 1) < EPS || fabs(nods(1, i) - 1) < EPS)
    {
      bnd.emplace_back(i);
    }
  }
}

class trig_poisson : public pde_problem
{
public:
  trig_poisson(const mati_t &tris, const matd_t &nods, const matd_t &mtrl)
      : nods_(nods), dim_(nods.cols())
  {
    ASSERT(tris.rows() == 3 && nods.rows() == 2 && mtrl.rows() == 2 && tris.cols() == mtrl.cols());

    matd_t ref = matd_t::Zero(2, 3);
    ref(0, 1) = ref(1, 2) = 1;
    matd_t Dm = ref.rightCols(2) - ref.leftCols(2);
    Dm = Dm.inverse();

    matd_t B = matd_t::Zero(3, 2);
    B(0, 0) = B(0, 1) = -1;
    B(1, 0) = B(2, 1) = 1;

    G_.resize(2 * tris.cols(), nods.cols());
    {
      vector<Triplet<double>> trips;
      for (size_t i = 0; i < tris.cols(); ++i)
      {
        matd_t Ds(2, 2);
        Ds.col(0) = nods.col(tris(1, i)) - nods.col(tris(0, i));
        Ds.col(1) = nods.col(tris(2, i)) - nods.col(tris(1, i));
        matd_t F = Ds * Dm;
        F = F.inverse();

        matd_t g = (B * F).transpose(); // 2x3
        for (size_t p = 0; p < g.rows(); ++p)
        {
          for (size_t q = 0; q < g.cols(); ++q)
          {
            trips.emplace_back(Triplet<double>(2 * i + p, tris(q, i), g(p, q)));
          }
        }
      }
      G_.setFromTriplets(trips.begin(), trips.end());
    }

    vecd_t cell_a = mtrl.row(0);
    const size_t unit_size = G_.rows() / tris.cols();
    A_.resize(G_.rows(), G_.rows());
    for (size_t i = 0; i < cell_a.size(); ++i)
    {
      for (size_t j = 0; j < unit_size; ++j)
      {
        A_.insert(unit_size * i + j, unit_size * i + j) = cell_a[i];
      }
    }
  }
  size_t dim() const
  {
    return dim_;
  }
  void LHS(const double *u, SparseMatrix<double> &A) const
  {
    A = G_.transpose() * A_ * G_;
  }
  void RHS(const double *u, VectorXd &b) const
  {
    b = VectorXd::Ones(dim());
  }

private:
  const matd_t &nods_;
  const size_t dim_;
  SparseMatrix<double> G_, A_;
};

extern "C"
{
  void tri_linear_elas_hes(double *, const double *, const double *, const double *, const double *, const double *);
}

class trig_elasticity : public pde_problem
{
public:
  trig_elasticity(const mati_t &tris, const matd_t &nods, const matd_t &mtrl)
      : tris_(tris), nods_(nods), lame_(mtrl), dim_(nods.size())
  {
    ASSERT(mtrl.rows() == 2 && tris.rows() == 3 && nods.rows() == 2 & tris.cols() == mtrl.cols());

    area_.resize(tris.cols());
    Dm_.resize(4, tris.cols());
#pragma omp parallel for
    for (size_t i = 0; i < tris_.cols(); ++i)
    {
      matd_t dm(2, 2);
      dm.col(0) = nods.col(tris(1, i)) - nods.col(tris(0, i));
      dm.col(1) = nods.col(tris(2, i)) - nods.col(tris(0, i));
      Map<Matrix2d>(&Dm_(0, i)) = dm.inverse();
      area_[i] = 0.5 * std::fabs(dm.determinant());
    }
  }
  size_t dim() const
  {
    return dim_;
  }
  void LHS(const double *u, SparseMatrix<double> &A) const
  {
    vector<Triplet<double>> trips;
    for (size_t i = 0; i < tris_.cols(); ++i)
    {
      double mu = lame_(0, i), lam = lame_(1, i);
      matd_t H = matd_t::Zero(6, 6);
      tri_linear_elas_hes(H.data(), nullptr, &Dm_(0, i), &area_[i], &lam, &mu);

      for (size_t p = 0; p < 6; ++p)
      {
        for (size_t q = 0; q < 6; ++q)
        {
          const size_t I = 2 * tris_(p / 2, i) + p % 2;
          const size_t J = 2 * tris_(q / 2, i) + q % 2;
          trips.emplace_back(Triplet<double>(I, J, H(p, q)));
        }
      }
    }

    A.resize(dim(), dim());
    A.reserve(trips.size());
    A.setFromTriplets(trips.begin(), trips.end());
  }
  void RHS(const double *u, VectorXd &b) const
  {
    b = VectorXd::Ones(dim());
  }

private:
  const size_t dim_;
  const mati_t &tris_;
  const matd_t &nods_;
  const matd_t &lame_;
  vecd_t area_;
  matd_t Dm_;
};

int main(int argc, char *argv[])
{
  boost::property_tree::ptree pt;
  read_cmdline(argc, argv, pt);

  const string outdir = pt.get<string>("outdir.value");
  const string mesh_type = "trig";
  const string prb_name = pt.get<string>("prb_name.value");
  const int RD = (prb_name == "laplacian") ? 1 : 2;
  spdlog::info("problem: {}, RD={}", prb_name, RD);

  const size_t coarsest_node_num = pt.get<size_t>("coarsest_node_num.value");
  spdlog::info("coarsest={}", coarsest_node_num);
  const size_t pts_num = pt.get<size_t>("num_samples.value");
  const string strategy = pt.get<string>("sample_strategy.value");
  const size_t sq_num = static_cast<size_t>(sqrt(1.0 * pts_num));

  //-> CHOL LEVELS
  std::vector<shared_ptr<chol_level>> levels;

  //-> Delaunay triangulation
  mati_t tris;
  matd_t nods;
  {
    // create hierarchy
    std::vector<Point> pts;
    if (strategy == "random")
    {
      std::mt19937 gen;
      std::uniform_real_distribution<double> dis(0.0, 1.0);

      //-> sample corners
      pts.emplace_back(Point(0, 0));
      pts.emplace_back(Point(1, 0));
      pts.emplace_back(Point(0, 1));
      pts.emplace_back(Point(1, 1));
      //-> sample boundary
      for (size_t i = 0; i < sq_num - 2; ++i)
      {
        double p = dis(gen);
        pts.push_back(Point(p, 0));
        p = dis(gen);
        pts.push_back(Point(p, 1));
        p = dis(gen);
        pts.push_back(Point(0, p));
        p = dis(gen);
        pts.push_back(Point(1, p));
      }
      //-> sample others
      size_t curr_size = pts.size();
      for (size_t i = curr_size; i < pts_num; ++i)
      {
        double x = dis(gen), y = dis(gen);
        pts.emplace_back(Point(x, y));
      }
    }
    else if (strategy == "regular")
    {
      const double dx = 1.0 / (sq_num - 1);
      for (size_t i = 0; i < sq_num; ++i)
      {
        for (size_t j = 0; j < sq_num; ++j)
        {
          pts.emplace_back(Point(i * dx, j * dx));
        }
      }
    }
    spdlog::info("sample number: {}", pts.size());

    // delaunay triangulation
    Delaunay dt;
    dt.insert(pts.begin(), pts.end());

    nods.resize(2, pts.size());
    std::map<Vertex_handle, size_t> v2i;
    size_t count = 0;
    for (Vertex_handle v : dt.finite_vertex_handles())
    {
      v2i.insert(std::make_pair(v, count));
      nods(0, count) = v->point().x();
      nods(1, count) = v->point().y();
      ++count;
    }
    ASSERT(pts.size() == count);

    spdlog::info("face number: {}", dt.number_of_faces());
    tris.resize(3, dt.number_of_faces());
    count = 0;
    for (Delaunay::Finite_faces_iterator fit = dt.finite_faces_begin(); fit != dt.finite_faces_end(); ++fit)
    {
      tris(0, count) = v2i[fit->vertex(0)];
      tris(1, count) = v2i[fit->vertex(1)];
      tris(2, count) = v2i[fit->vertex(2)];
      ++count;
    }
    ASSERT(count == dt.number_of_faces());

    volume_calculator vc(tris, nods, mesh_type);
    spdlog::info("DT mesh volume: {}", vc.compute());

    chol_hierarchy builder(tris, nods, mesh_type);
    ASSERT(coarsest_node_num > 0);
    builder.build(levels, coarsest_node_num, RD);
    {
      for (size_t l = 0; l < levels.size(); ++l)
      {
        char outf[256];
        sprintf(outf, "%s/point-level-%zu.vtk", outdir.c_str(), l);
        point_write_to_vtk(outf, levels[l]->nods_);
      }
    }
  }
  const auto &FL = levels.back();

  //-> generate material
  const string mtr_name = pt.get<string>("mtr_name.value");
  const double maxE = pt.get<double>("max_E.value");
  const double minE = pt.get<double>("min_E.value");
  spdlog::info("max min E: {} {}", maxE, minE);
  matd_t FL_mtr;
  FL_mtr = generate_random_material(FL->cell_, mtr_name, maxE, minE);
  {
    char outf[256];
    sprintf(outf, "%s/vis-mtr.vtk", outdir.c_str());
    tri_mesh_write_to_vtk(outf, FL->nods_, FL->cell_, &FL_mtr, "CELL");
  }

  vector<size_t> bnd_node;
  extr_boundary_node(FL->cell_, FL->nods_, bnd_node);
  spdlog::info("boundary node number: {}", bnd_node.size());
  cout << "=======================================================" << endl;

  /////////////// solve problems ///////////////
  std::shared_ptr<pde_problem> prb;
  if (prb_name == "laplacian")
  {
    prb = std::make_shared<trig_poisson>(FL->cell_, FL->nods_, FL_mtr);
  }
  else if (prb_name == "elasticity")
  {
    prb = std::make_shared<trig_elasticity>(FL->cell_, FL->nods_, FL_mtr);
  }
  else
  {
    bool problem_is_not_supported = false;
    ASSERT(problem_is_not_supported);
  }

  SparseMatrix<double> A;
  VectorXd b, init_x = VectorXd::Zero(prb->dim());
  prb->LHS(init_x.data(), A);
  prb->RHS(init_x.data(), b);
  //-> impose Dirichlet boundary conditions via soft penalties
  const double wp = pt.get<double>("wp.value");
  for (const auto j : bnd_node)
  {
    auto ptr = A.outerIndexPtr();
    auto ind = A.innerIndexPtr();
    auto val = A.valuePtr();
    for (size_t d = 0; d < RD; ++d)
    {
      const size_t col = RD * j + d;
      for (size_t iter = ptr[col]; iter < ptr[col + 1]; ++iter)
      {
        if (col == ind[iter])
        {
          val[iter] += wp;
          break;
        }
      }
    }
  }
  // vis_spmat(outdir + "/A.png", A);

  high_resolution_timer hrt;

  //-> init multiscale preconditioner
  std::shared_ptr<preconditioner> cg_prec;
  const string prec_name = pt.get<string>("precond.value");
  hrt.start();
  if (prec_name == "ichol")
  {
    cg_prec = std::make_shared<ichol_precond>(levels, pt);
  }
  else if (prec_name == "amg")
  {
    const size_t nrelax = 1, cycle_maxits = 1;
    cg_prec = std::make_shared<amg_precon>(nrelax, cycle_maxits);
  }
  else
  {
    bool invalid_preconditioner_type = 0;
    ASSERT(invalid_preconditioner_type);
  }
  hrt.stop();
  double pre_time = hrt.duration() / 1000;
  spdlog::info("precompute time: {}", pre_time);

  precond_cg_solver pcg(cg_prec);

  hrt.start();
  pcg.analyse_pattern(A);
  hrt.stop();
  double syb_time = hrt.duration() / 1000;
  spdlog::info("symbolic time: {}\n", syb_time);

  hrt.start();
  pcg.factorize(A, false);
  hrt.stop();
  double fac_time = hrt.duration() / 1000;
  spdlog::info("fac time: {}", fac_time);

  hrt.start();
  VectorXd u = pcg.solve(b);
  hrt.stop();
  double slv_time = hrt.duration() / 1000;
  spdlog::info("slv time: {}", slv_time);
  spdlog::info("all time: {}", fac_time + slv_time);
  spdlog::info("pcg iter: {}", pcg.m_iters);
  {
    const matd_t &field = Map<matd_t>(u.data(), RD, u.size() / RD);
    tri_mesh_write_to_vtk(string(outdir + "/result.vtk").c_str(), FL->nods_, FL->cell_, &field, "POINT");
  }

  //-> write configuration
  ofstream json_ofs(outdir + "/config.json");
  boost::property_tree::write_json(json_ofs, pt, true);
  json_ofs.close();

  cout << "[INFO] done" << endl;
  return 0;
}
