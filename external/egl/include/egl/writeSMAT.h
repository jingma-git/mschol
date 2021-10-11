#pragma once

#include <Eigen/Sparse>
#include <fstream>
#include <vector>

namespace egl
{

    template <typename _Scalar, int _Options, typename _StorageIndex>
    void writeSMAT(const std::string fname,
                   const Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> &m)
    {
        using namespace std;
        using namespace Eigen;

        typedef Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> spmat_t;
        typedef _StorageIndex IND;
        typedef _Scalar T;

        typedef Triplet<int> Trip;
        std::vector<Trip> res;

        spmat_t A = m;
        int sz = A.nonZeros();
        A.makeCompressed();

        fstream writeFile;
        writeFile.open(fname.c_str(), ios::binary | ios::out);

        if (writeFile.is_open())
        {
            IND rows, cols, nnzs, outS, innS;
            rows = A.rows();
            cols = A.cols();
            nnzs = A.nonZeros();
            outS = A.outerSize();
            innS = A.innerSize();

            writeFile.write((const char *)&(rows), sizeof(IND));
            writeFile.write((const char *)&(cols), sizeof(IND));
            writeFile.write((const char *)&(nnzs), sizeof(IND));
            writeFile.write((const char *)&(innS), sizeof(IND));
            writeFile.write((const char *)&(outS), sizeof(IND));

            writeFile.write((const char *)(A.valuePtr()), sizeof(T) * A.nonZeros());
            writeFile.write((const char *)(A.outerIndexPtr()), sizeof(IND) * A.outerSize());
            writeFile.write((const char *)(A.innerIndexPtr()), sizeof(IND) * A.nonZeros());

            writeFile.close();
        }
    }
}