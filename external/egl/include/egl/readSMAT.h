#pragma once

#include <Eigen/Sparse>
#include <fstream>

namespace egl
{
    template <typename T, int whatever, typename IND>
    void readSMAT(const std::string fname, Eigen::SparseMatrix<T, whatever, IND> &m)
    {
        using namespace std;

        fstream readFile;
        readFile.open(fname.c_str(), ios::binary | ios::in);
        if (readFile.is_open())
        {
            IND rows, cols, nnz, inSz, outSz;
            readFile.read((char *)&rows, sizeof(IND));
            readFile.read((char *)&cols, sizeof(IND));
            readFile.read((char *)&nnz, sizeof(IND));
            readFile.read((char *)&inSz, sizeof(IND));
            readFile.read((char *)&outSz, sizeof(IND));

            m.resize(rows, cols);
            m.makeCompressed();
            m.resizeNonZeros(nnz);

            readFile.read((char *)(m.valuePtr()), sizeof(T) * nnz);
            readFile.read((char *)(m.outerIndexPtr()), sizeof(IND) * outSz);
            readFile.read((char *)(m.innerIndexPtr()), sizeof(IND) * nnz);

            m.finalize();
            readFile.close();

        } // file is open
    }
}