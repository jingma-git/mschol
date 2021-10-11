#pragma once

#include <opencv2/opencv.hpp>
#include <Eigen/Sparse>
#include <string>
namespace mschol
{

    template <typename _Scalar, int _Options, typename _StorageIndex>
    void vis_spmat(const std::string img_path, const Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex> &A)
    {
        if (A.rows() > 2048 && A.cols() > 2048)
            return;
        // assert(A.rows() < 2048 && A.cols() < 2048 && "vis_spmat: A is too large to be visualized!");
        cv::Mat img(A.rows(), A.cols(), CV_8UC1, 255);
        for (int j = 0; j < A.outerSize(); ++j)
        {
            for (typename Eigen::SparseMatrix<_Scalar, _Options, _StorageIndex>::InnerIterator it(A, j);
                 it; ++it)
            {
                img.at<uchar>(it.row(), it.col()) = 0;
            }
        }
        cv::imwrite(img_path, img);
    }
}