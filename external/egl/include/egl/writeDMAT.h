#pragma once
#include <string>
#include <cstdio>
#include <Eigen/Core>

namespace egl
{
    template <typename DerivedW>
    bool writeDMAT(
        const std::string file_name,
        const Eigen::MatrixBase<DerivedW> &W,
        const bool ascii)
    {
        FILE *fp = fopen(file_name.c_str(), "wb");
        if (fp == NULL)
        {
            fprintf(stderr, "IOError: writeDMAT() could not open %s...", file_name.c_str());
            return false;
        }
        if (ascii)
        {
            // first line contains number of rows and number of columns
            fprintf(fp, "%d %d\n", (int)W.cols(), (int)W.rows());
            // Loop over columns slowly
            for (int j = 0; j < W.cols(); j++)
            {
                // loop over rows (down columns) quickly
                for (int i = 0; i < W.rows(); i++)
                {
                    fprintf(fp, "%0.17lg\n", (double)W(i, j));
                }
            }
        }
        else
        {
            // write header for ascii
            fprintf(fp, "0 0\n");
            // first line contains number of rows and number of columns
            fprintf(fp, "%d %d\n", (int)W.cols(), (int)W.rows());
            // reader assumes the binary part is double precision
            Eigen::MatrixXd Wd = W.template cast<double>();
            fwrite(Wd.data(), sizeof(double), Wd.size(), fp);
        }
        fclose(fp);
        return true;
    }
}
