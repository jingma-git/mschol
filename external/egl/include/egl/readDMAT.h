#pragma once
#include <cstdio>
#include <iostream>
#include <cassert>
#include <memory>
#include <Eigen/Core>
namespace egl
{

    // Static helper method reads the first to elements in the given file
    // Inputs:
    //   fp  file pointer of .dmat file that was just opened
    // Outputs:
    //   num_rows  number of rows
    //   num_cols number of columns
    // Returns
    //   0  success
    //   1  did not find header
    //   2  bad num_cols
    //   3  bad num_rows
    //   4  bad line ending
    inline int readDMAT_read_header(FILE *fp, int &num_rows, int &num_cols)
    {
        // first line contains number of rows and number of columns
        int res = fscanf(fp, "%d %d", &num_cols, &num_rows);
        if (res != 2)
        {
            return 1;
        }
        // check that number of columns and rows are sane
        if (num_cols < 0)
        {
            fprintf(stderr, "IOError: readDMAT() number of columns %d < 0\n", num_cols);
            return 2;
        }
        if (num_rows < 0)
        {
            fprintf(stderr, "IOError: readDMAT() number of rows %d < 0\n", num_rows);
            return 3;
        }
        // finish reading header
        char lf;

        if (fread(&lf, sizeof(char), 1, fp) != 1 || !(lf == '\n' || lf == '\r'))
        {
            fprintf(stderr, "IOError: bad line ending in header\n");
            return 4;
        }

        return 0;
    }

    template <typename DerivedW>
    bool readDMAT(const std::string file_name,
                  Eigen::PlainObjectBase<DerivedW> &W)
    {
        FILE *fp = fopen(file_name.c_str(), "rb");
        if (fp == NULL)
        {
            fprintf(stderr, "IOError: readDMAT() could not open %s...\n", file_name.c_str());
            return false;
        }
        int num_rows, num_cols;
        int head_success = readDMAT_read_header(fp, num_rows, num_cols);
        if (head_success != 0)
        {
            if (head_success == 1)
            {
                fprintf(stderr,
                        "IOError: readDMAT() first row should be [num cols] [num rows]...\n");
            }
            fclose(fp);
            return false;
        }

        // Resize output to fit matrix, only if non-empty since this will trigger an
        // error on fixed size matrices before reaching binary data.
        bool empty = num_rows == 0 || num_cols == 0;
        if (!empty)
        {
            W.resize(num_rows, num_cols);
        }

        // Loop over columns slowly
        for (int j = 0; j < num_cols; j++)
        {
            // loop over rows (down columns) quickly
            for (int i = 0; i < num_rows; i++)
            {
                double d;
                if (fscanf(fp, " %lg", &d) != 1)
                {
                    fclose(fp);
                    fprintf(
                        stderr,
                        "IOError: readDMAT() bad format after reading %d entries\n",
                        j * num_rows + i);
                    return false;
                }
                W(i, j) = d;
            }
        }

        // Try to read header for binary part
        head_success = readDMAT_read_header(fp, num_rows, num_cols);
        if (head_success == 0)
        {
            assert(W.size() == 0);
            // Resize for output
            W.resize(num_rows, num_cols);
            std::unique_ptr<double[]> Wraw(new double[num_rows * num_cols]);
            fread(Wraw.get(), sizeof(double), num_cols * num_rows, fp);
            // Loop over columns slowly
            for (int j = 0; j < num_cols; j++)
            {
                // loop over rows (down columns) quickly
                for (int i = 0; i < num_rows; i++)
                {
                    W(i, j) = Wraw[j * num_rows + i];
                }
            }
        }
        else
        {
            // we skipped resizing before in case there was binary data
            if (empty)
            {
                // This could trigger an error if using fixed size matrices.
                W.resize(num_rows, num_cols);
            }
        }

        fclose(fp);
        return true;
    }
}