#include <fstream>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <spdlog/spdlog.h>

#include "src/vtk.h"
#include "src/io.h"
#include "src/chol_hierarchy.h"
#include "src/volume.h"
#include "src/cholesky_level.h"
#include "src/cholesky.h"
#include "src/ptree.h"
#include "src/util.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Delaunay;
typedef K::Point_3 Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Cell_handle Cell_handle;

using namespace std;
using namespace mschol;
using namespace Eigen;

static void extract_boundary_nodes(const mati_t &tets, const matd_t &nods,
                                   const string &prb_name,
                                   std::vector<size_t> &bnd_nodes) {
  const double EPS = 1e-10;
  for (size_t i = 0; i < nods.cols(); ++i) {
    if ( prb_name == "elasticity" ) {
      if ( fabs(nods(0, i)-0) < EPS ) {
        bnd_nodes.emplace_back(i);
      }
    } else if ( prb_name == "laplacian" ) {
      if ( fabs(nods(0, i)-0) < EPS || fabs(nods(0, i)-1) < EPS ||
           fabs(nods(1, i)-0) < EPS || fabs(nods(1, i)-1) < EPS ||
           fabs(nods(2, i)-0) < EPS || fabs(nods(2, i)-1) < EPS ) {
        bnd_nodes.emplace_back(i);
      }                                                          
    }
  }
}

class tet_poisson : public pde_problem
{
 public:
  tet_poisson(const mati_t &tets, const matd_t &nods, const matd_t &mtrl)
      : nods_(nods), dim_(nods.cols()) {
    ASSERT(tets.rows() == 4 && nods.rows() == 3 && mtrl.rows() == 2 && tets.cols() == mtrl.cols());

    matd_t ref = matd_t::Zero(3, 4);
    ref.rightCols(3).setIdentity();
    matd_t Dm(3, 3);
    Dm.col(0) = ref.col(1)-ref.col(0);
    Dm.col(1) = ref.col(2)-ref.col(0);
    Dm.col(2) = ref.col(3)-ref.col(0);
    Dm = Dm.inverse();

    matd_t B = matd_t::Zero(4, 3);
    B.row(0).setOnes();
    B.row(0) *= -1;
    B.bottomRows(3).setIdentity();
    
    G_.resize(3*tets.cols(), nods.cols());
    {
      vector<Triplet<double>> trips;
      for (size_t i = 0; i < tets.cols(); ++i) {
        matd_t Ds(3, 3);
        Ds.col(0) = nods.col(tets(1, i))-nods.col(tets(0, i));
        Ds.col(1) = nods.col(tets(2, i))-nods.col(tets(0, i));
        Ds.col(2) = nods.col(tets(3, i))-nods.col(tets(0, i));
        matd_t F = Ds*Dm;
        F = F.inverse();

        matd_t g = (B*F).transpose(); // 3x4
        for (size_t p = 0; p < g.rows(); ++p) {
          for (size_t q = 0; q < g.cols(); ++q) {
            trips.emplace_back(Triplet<double>(3*i+p, tets(q, i), g(p, q)));
          }
        }
      }
      G_.setFromTriplets(trips.begin(), trips.end());
    }

    vecd_t cell_a = mtrl.row(0);
    const size_t unit_size = G_.rows()/tets.cols();
    A_.resize(G_.rows(), G_.rows());
    for (size_t i = 0; i < cell_a.size(); ++i) {
      for (size_t j = 0; j < unit_size; ++j) {
        A_.insert(unit_size*i+j, unit_size*i+j) = cell_a[i];
      }
    }
  }
  size_t dim() const {
    return dim_;
  }
  void LHS(const double *u, SparseMatrix<double> &A) const {
    A = G_.transpose()*A_*G_;
  }
  void RHS(const double *u, VectorXd &b) const {
    b = VectorXd::Ones(dim());
  }
  
 private:
  const matd_t &nods_;
  const size_t dim_;
  SparseMatrix<double> G_, A_;
};

extern "C" {
  void tet_linear_elas_hes(double*, const double*, const double*, const double*, const double*, const double*);
}

class tet_elasticity : public pde_problem
{
 public:
  tet_elasticity(const mati_t &tets, const matd_t &nods, const matd_t &mtrl)
      : dim_(nods.size()), tets_(tets), nods_(nods), lame_(mtrl) {
    ASSERT(tets.rows() == 4 && nods.rows() == 3 && mtrl.rows() == 2 && tets.cols() == mtrl.cols());
    vol_.resize(1, tets_.cols());
    Dm_.resize(9, tets_.cols());

    #pragma omp parallel for
    for (size_t i = 0; i < tets_.cols(); ++i) {
      matd_t dm(3, 3);
      dm.col(0) = nods.col(tets(1, i))-nods.col(tets(0, i));
      dm.col(1) = nods.col(tets(2, i))-nods.col(tets(0, i));
      dm.col(2) = nods.col(tets(3, i))-nods.col(tets(0, i));
      vol_[i] = fabs(dm.determinant())/6.0;
      Map<Matrix3d>(&Dm_(0, i)) = dm.inverse();
    }    
  }
  size_t dim() const {
    return dim_;
  }
  void LHS(const double *u, SparseMatrix<double> &A) const {
    vector<Triplet<double>> trips;
    #pragma omp parallel for
    for (size_t i = 0; i < tets_.cols(); ++i) {
      matd_t H = matd_t::Zero(12, 12);
      tet_linear_elas_hes(H.data(), nullptr, &Dm_(0, i), &vol_[i], &lame_(1, i), &lame_(0, i));

      vector<Triplet<double>> local_hes;
      local_hes.reserve(H.size());
      for (size_t p = 0; p < 12; ++p) {
        for (size_t q = 0; q < 12; ++q) {
          const size_t I = 3*tets_(p/3, i)+p%3;
          const size_t J = 3*tets_(q/3, i)+q%3;
          local_hes.emplace_back(Triplet<double>(I, J, H(p, q)));
        }
      } 

      #pragma omp critical
      {
        trips.insert(trips.end(), local_hes.begin(), local_hes.end());
      }
    }

    A.resize(dim(), dim());
    A.reserve(trips.size());
    A.setFromTriplets(trips.begin(), trips.end());
  }
  void RHS(const double *u, VectorXd &b) const {
    b = VectorXd::Ones(dim());
  }
  
 private:
  const size_t dim_;
  const mati_t &tets_;
  const matd_t &nods_;
  const matd_t &lame_;
  vecd_t vol_;
  matd_t Dm_;
};

int main(int argc, char *argv[])
{    
  boost::property_tree::ptree pt;
  read_cmdline(argc, argv, pt);

  const string outdir = pt.get<string>("outdir.value");
  const string mesh_type = "tets";
  const string prb_name = pt.get<string>("prb_name.value");
  const int RD = (prb_name == "laplacian") ? 1 : 3;
  spdlog::info("problem: {}, RD={}", prb_name, RD);

  const size_t coarsest_node_num = pt.get<size_t>("coarsest_node_num.value");
  spdlog::info("coarsest={}", coarsest_node_num);  
  const size_t pts_num = pt.get<size_t>("num_samples.value");
  const string strategy = pt.get<string>("sample_strategy.value");
  const size_t sq_num = static_cast<size_t>(cbrt(1.0*pts_num));

  std::vector<shared_ptr<chol_level>> levels;
 
  //-> Delaunay triangulation
  mati_t tets; matd_t nods;
  {
    std::vector<Point> pts;
    if ( strategy == "random" ) {
      std::mt19937 gen;
      std::uniform_real_distribution<double> dis(0.0, 1.0);
      //-> sample corners
      pts.emplace_back(Point(0, 0, 0));
      pts.emplace_back(Point(1, 0, 0));
      pts.emplace_back(Point(0, 1, 0));
      pts.emplace_back(Point(1, 1, 0));
      pts.emplace_back(Point(0, 0, 1));
      pts.emplace_back(Point(1, 0, 1));
      pts.emplace_back(Point(0, 1, 1));
      pts.emplace_back(Point(1, 1, 1));
      //-> sample boundary edges
      for (size_t i = 0; i < sq_num-2; ++i) {
        double p;
        p = dis(gen); pts.emplace_back(Point(p, 0, 0));
        p = dis(gen); pts.emplace_back(Point(p, 1, 0));
        p = dis(gen); pts.emplace_back(Point(p, 0, 1));
        p = dis(gen); pts.emplace_back(Point(p, 1, 1));
        p = dis(gen); pts.emplace_back(Point(0, p, 0));
        p = dis(gen); pts.emplace_back(Point(1, p, 0));
        p = dis(gen); pts.emplace_back(Point(0, p, 1));
        p = dis(gen); pts.emplace_back(Point(1, p, 1));
        p = dis(gen); pts.emplace_back(Point(0, 0, p));
        p = dis(gen); pts.emplace_back(Point(1, 0, p));
        p = dis(gen); pts.emplace_back(Point(0, 1, p));
        p = dis(gen); pts.emplace_back(Point(1, 1, p));
      }
      //-> sample boundary faces
      for (size_t i = 0; i < (sq_num-2)*(sq_num-2); ++i) {
        double x, y;
        x = dis(gen); y = dis(gen); pts.emplace_back(Point(x, y, 0));
        x = dis(gen); y = dis(gen); pts.emplace_back(Point(x, y, 1));
        x = dis(gen); y = dis(gen); pts.emplace_back(Point(x, 0, y));
        x = dis(gen); y = dis(gen); pts.emplace_back(Point(x, 1, y));
        x = dis(gen); y = dis(gen); pts.emplace_back(Point(0, x, y));
        x = dis(gen); y = dis(gen); pts.emplace_back(Point(1, x, y));
      }      
      //-> sample others
      size_t curr_size = pts.size();
      for (size_t i = curr_size; i < pts_num; ++i) {
        double x = dis(gen), y = dis(gen), z = dis(gen);
        pts.emplace_back(Point(x, y, z));
      }
    } else if ( strategy == "regular" ) {
      const double dx = 1.0/(sq_num-1);
      for (size_t i = 0; i < sq_num; ++i) {
        for (size_t j = 0; j < sq_num; ++j) {
          for (size_t k = 0; k < sq_num; ++k) {
            pts.emplace_back(Point(i*dx, j*dx, k*dx));
          }
        }
      }
    }
    spdlog::info("sample number: {}", pts.size());    
    
    // delaunay triangulation
    Delaunay dt;
    dt.insert(pts.begin(), pts.end());

    nods.resize(3, pts.size());
    std::map<Vertex_handle, size_t> v2i;
    size_t count = 0;
    for (Vertex_handle v : dt.finite_vertex_handles()) {
      v2i.insert(std::make_pair(v, count));
      nods(0, count) = v->point().x();
      nods(1, count) = v->point().y();
      nods(2, count) = v->point().z();
      ++count;
    }
    ASSERT(pts.size() == count);

    spdlog::info("face number: {}", dt.number_of_finite_cells());
    tets.resize(4, dt.number_of_finite_cells());
    count = 0;
    for (Delaunay::Finite_cells_iterator fit = dt.finite_cells_begin(); fit != dt.finite_cells_end(); ++fit) {
      tets(0, count) = v2i[fit->vertex(0)];
      tets(1, count) = v2i[fit->vertex(1)];
      tets(2, count) = v2i[fit->vertex(2)];
      tets(3, count) = v2i[fit->vertex(3)];
      ++count;
    }
    ASSERT(count == dt.number_of_finite_cells());

    volume_calculator vc(tets, nods, mesh_type);
    spdlog::info("DT mesh volume: {}", vc.compute());

    chol_hierarchy builder(tets, nods, mesh_type);
    ASSERT(coarsest_node_num > 0);
    builder.build(levels, coarsest_node_num, RD);
    {
      for (size_t l = 0; l < levels.size(); ++l) {
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
  matd_t FL_mtr = generate_random_material(FL->cell_, mtr_name, maxE, minE);
  {
    char outf[256];
    sprintf(outf, "%s/vis-mtr.vtk", outdir.c_str());
    tet_mesh_write_to_vtk(outf, FL->nods_, FL->cell_, &FL_mtr, "CELL");
  }

  vector<size_t> bnd_node;
  extract_boundary_nodes(FL->cell_, FL->nods_, prb_name, bnd_node);    
  spdlog::info("boundary node number: {}", bnd_node.size());
  cout << "=======================================================" << endl;

  /////////////// solve problems ///////////////    
  std::shared_ptr<pde_problem> prb;
  if ( prb_name == "laplacian" ) {
    prb = std::make_shared<tet_poisson>(FL->cell_, FL->nods_, FL_mtr);
  } else if ( prb_name == "elasticity" ) {
    prb = std::make_shared<tet_elasticity>(FL->cell_, FL->nods_, FL_mtr);
  } else {
    bool problem_is_not_supported = false;
    ASSERT(problem_is_not_supported);
  }

  SparseMatrix<double> A;
  VectorXd b, init_x = VectorXd::Zero(prb->dim());
  prb->LHS(init_x.data(), A);
  prb->RHS(init_x.data(), b);
  //-> impose Dirichlet boundary conditions via soft penalties
  const double wp = pt.get<double>("wp.value");
  for (const auto j : bnd_node) {
    auto ptr = A.outerIndexPtr();
    auto ind = A.innerIndexPtr();
    auto val = A.valuePtr();
    for (size_t d = 0; d < RD; ++d) {
      const size_t col = RD*j+d;
      for (size_t iter = ptr[col]; iter < ptr[col+1]; ++iter) {
        if ( col == ind[iter] ) {
          val[iter] += wp;
          break;            
        }
      }
    }
  }

  high_resolution_timer hrt;
    
  //-> init multiscale preconditioner
  std::shared_ptr<preconditioner> cg_prec;    
  const string prec_name = pt.get<string>("precond.value");
  hrt.start();
  if ( prec_name == "ichol" ) {
    cg_prec = std::make_shared<ichol_precond>(levels, pt);
  } else if ( prec_name == "amg" ) {
    const size_t nrelax = 1, cycle_maxits = 1;
    cg_prec = std::make_shared<amg_precon>(nrelax, cycle_maxits);
  } else {
    bool invalid_preconditioner_type = 0;
    ASSERT(invalid_preconditioner_type);
  }
  hrt.stop();
  double pre_time = hrt.duration();
  spdlog::info("precompute time: {}", pre_time);

  precond_cg_solver pcg(cg_prec);

  hrt.start();
  pcg.analyse_pattern(A);
  hrt.stop();
  double syb_time = hrt.duration()/1000;
  spdlog::info("symbolic time: {}\n", syb_time);

  hrt.start();
  pcg.factorize(A, false);
  hrt.stop();
  double fac_time = hrt.duration()/1000;
  spdlog::info("fac time: {}", fac_time);
    
  hrt.start();
  VectorXd u = pcg.solve(b);
  hrt.stop();
  double slv_time = hrt.duration()/1000;
  spdlog::info("slv time: {}", slv_time);
  spdlog::info("all time: {}", fac_time+slv_time);
  spdlog::info("pcg iter: {}", pcg.m_iters);
  {
    const matd_t &field = Map<matd_t>(u.data(), RD, u.size()/RD);
    tet_mesh_write_to_vtk(string(outdir+"/result.vtk").c_str(), FL->nods_, FL->cell_, &field, "POINT");
  }  
    
  //-> write configuration
  ofstream json_ofs(outdir+"/config.json");
  boost::property_tree::write_json(json_ofs, pt, true);
  json_ofs.close();

  cout << "[INFO] done" << endl;
  return 0;
}
