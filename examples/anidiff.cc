#include <fstream>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/draw_triangulation_2.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include "src/io.h"
#include "src/chol_hierarchy.h"
#include "src/volume.h"
#include "src/cholesky_level.h"
#include "src/cholesky.h"
#include "src/ptree.h"
#include "src/util.h"

using namespace std;
using namespace mschol;
using namespace Eigen;

typedef double Float;

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

void spIdentity(int n, Eigen::SparseMatrix<Float> &I)
{
    std::vector<Eigen::Triplet<Float>> data(n);
    for (int i = 0; i < n; ++i)
    {
        data.emplace_back(i, i, 1.0);
    }
    I.resize(n, n);
    I.setFromTriplets(data.begin(), data.end());
}

void saveImg(const vecd_t &x, int rows, int cols, const std::string fname)
{
    Eigen::Array<unsigned char, -1, -1> bits = (x).cast<unsigned char>();
    cv::Mat img(rows, cols, CV_8UC1, bits.data());
    cv::imwrite(fname, img);
}

class img_poisson : public pde_problem
{
public:
    // algorithm parameters
    Float lambda = 1000.0;
    const Float K = 10;
    const Float K2 = 1 / K / K;

public:
    img_poisson(const mati_t &cells, const matd_t &nods, int rows, int cols)
        : cels_(cells), nods_(nods), rows_(rows), cols_(cols), dim_(rows * cols)
    {
        spIdentity(dim_, I_);
    }

    size_t dim() const
    {
        return dim_;
    }

    void LHS(const Float *u, Eigen::SparseMatrix<Float> &A) const
    {
        int rows = rows_;
        int cols = cols_;
        matd_t C(rows, cols);
        grad(u, rows, cols, C);
        std::vector<Eigen::Triplet<Float>> coeffs;
        for (int j = 0; j < cols; ++j)
        {
            for (int i = 0; i < rows; ++i)
            {
                int id = i * cols + j;

                Float Cp = C(i, j);
                Float Cn = (i == 0) ? 0 : C(i - 1, j);
                Float Cs = (i == rows - 1) ? 0 : C(i + 1, j);
                Float Ce = (j == cols - 1) ? 0 : C(i, j + 1);
                Float Cw = (j == 0) ? 0 : C(i, j - 1);

                insertCoefficient(id, i - 1, j, Cn + Cp, coeffs, rows, cols);
                insertCoefficient(id, i + 1, j, Cs + Cp, coeffs, rows, cols);
                insertCoefficient(id, i, j - 1, Cw + Cp, coeffs, rows, cols);
                insertCoefficient(id, i, j + 1, Ce + Cp, coeffs, rows, cols);
            }
        }
        Eigen::SparseMatrix<Float> L(dim_, dim_);
        L.setFromTriplets(coeffs.begin(), coeffs.end());
        A = I_ - lambda * L;
    }

    void RHS(const Float *u, vecd_t &b) const
    {
        b = Eigen::Map<const vecd_t>(u, dim_, 1);
    }

private:
    void grad(const Float *ptr, int rows, int cols, matd_t &C) const // d: diffusion coeff
    {
        cv::Mat It(rows, cols, sizeof(Float) == 4 ? CV_32F : CV_64F, const_cast<Float *>(ptr)); // image at time step t
        C = matd_t::Zero(rows, cols);
        cv::Mat dx, dy;

        cv::Sobel(It, dx, sizeof(Float) == 4 ? CV_32F : CV_64F, 1, 0, 3); // gradient along x
        cv::Sobel(It, dy, sizeof(Float) == 4 ? CV_32F : CV_64F, 0, 1, 3); // gradient along y
        for (int i = 0; i < It.rows; ++i)
            for (int j = 0; j < It.cols; ++j)
            {
                Float gx = dx.at<Float>(i, j), gy = dy.at<Float>(i, j);
                Float c;
                if (i == 0 || i == It.rows - 1 || j == 0 || j == It.cols - 1)
                    c = 0; // no diffusion on boundary
                else
                {
                    // c = std::exp(-(gx * gx + gy * gy) * K2);
                    c = 1.0 / (1.0 + (gx * gx + gy * gy) * K2);
                }
                C(i, j) = c;
            }
    }

    void insertCoefficient(int id, int i, int j, Float w, std::vector<Eigen::Triplet<Float>> &coeffs,
                           int rows, int cols) const
    {
        int id1 = i * cols + j;
        if (i >= 0 && i < rows && j >= 0 && j < cols)
        {
            // cout << id << "," << id1 << ": " << w << endl;
            coeffs.emplace_back(id, id1, w); // expensive compared with explicit counterpart
            coeffs.emplace_back(id, id, -w);
        }
    }

private:
    const mati_t &cels_;
    const matd_t &nods_;
    int rows_, cols_, dim_;
    Eigen::SparseMatrix<Float> I_;
};

int main(int argc, char *argv[])
{
    boost::property_tree::ptree pt;
    read_cmdline(argc, argv, pt);
    const std::string inpath = pt.get<string>("inpath.value");
    const string outdir = pt.get<string>("outdir.value");
    const string mesh_type = "trig";
    const string prb_name = pt.get<string>("prb_name.value");
    const int RD = (prb_name == "laplacian") ? 1 : 2;
    spdlog::info("problem: {}, RD={}", prb_name, RD);
    const size_t coarsest_node_num = pt.get<size_t>("coarsest_node_num.value");
    spdlog::info("coarsest={}", coarsest_node_num);
    const int imgh = pt.get<int>("imgh.value");
    const int imgw = pt.get<int>("imgw.value");

    cv::Mat img = cv::imread(inpath, cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, cv::Size(imgh, imgw)); // 5, 8
    cv::Mat I0(img.rows, img.cols, sizeof(Float) == 4 ? CV_32F : CV_64F);
    img.convertTo(I0, sizeof(Float) == 4 ? CV_32F : CV_64F); // 4 bytes == 32 bits
    int num_nods = I0.rows * I0.cols;
    int num_cels = (I0.rows - 1) * (I0.cols - 1) * 2;
    spdlog::info("precision:{} rows:{}, cols:{}, num_nods:{} num_cels:{}", sizeof(Float), I0.rows, I0.cols, num_nods, num_cels);

    //-> CHOL LEVELS
    std::vector<shared_ptr<chol_level>> levels;
    mati_t cels;
    matd_t nods;
    {
        nods.resize(2, num_nods);
        cels.resize(3, num_cels);
        int vidx = 0, fidx = 0;
        for (int i = 0; i < I0.rows; ++i)
        {
            for (int j = 0; j < I0.cols; ++j)
            {
                nods(0, vidx) = i;
                nods(1, vidx) = j;
                ++vidx;
            }
        }
        for (int i = 0; i < I0.rows - 1; ++i)
        {
            for (int j = 0; j < I0.cols - 1; ++j)
            {
                int v0 = i * I0.cols + j;
                int v1 = i * I0.cols + (j + 1);
                int v2 = (i + 1) * I0.cols + j;
                int v3 = (i + 1) * I0.cols + (j + 1);
                cels(0, fidx) = v0;
                cels(1, fidx) = v3;
                cels(2, fidx) = v2;
                ++fidx;
                cels(0, fidx) = v0;
                cels(1, fidx) = v1;
                cels(2, fidx) = v3;
                ++fidx;
            }
        }

        chol_hierarchy builder(cels, nods, mesh_type);
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

    /////////////// solve problems ///////////////
    std::shared_ptr<pde_problem> prb;
    prb = std::make_shared<img_poisson>(FL->cell_, FL->nods_, I0.rows, I0.cols);

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

    VectorXd b;
    prb->RHS((Float *)I0.data, b);
    saveImg(b, I0.rows, I0.cols, outdir + "ini.png");
    SparseMatrix<double> A;
    prb->LHS((Float *)I0.data, A);

    precond_cg_solver pcg(cg_prec);

    hrt.start();
    pcg.analyse_pattern(A);
    hrt.stop();
    double syb_time = hrt.duration() / 1000;
    spdlog::info("symbolic time: {}\n", syb_time);

    SparseMatrix<double> L;
    prb->LHS((Float *)I0.data, L);
    hrt.start();
    pcg.factorize(L, false);
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
    saveImg(u, I0.rows, I0.cols, outdir + "iter0.png");
    if (A.rows() < 80)
    {
        for (int k = 0; k < A.outerSize(); ++k)
        {
            for (Eigen::SparseMatrix<Float>::InnerIterator it(A, k); it; ++it)
            {
                cout << it.row() << ", " << it.col() << " " << it.value() << endl;
            }
        }
        cout << "b\n"
             << b.transpose() << endl;
    }

    //-> write configuration
    ofstream json_ofs(outdir + "/config.json");
    boost::property_tree::write_json(json_ofs, pt, true);
    json_ofs.close();

    cout << "[INFO] done" << endl;
    return 0;
}
