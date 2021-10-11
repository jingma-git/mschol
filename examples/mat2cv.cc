#include <opencv2/opencv.hpp>
#include <egl/readDMAT.h>
#include <cstdlib>

void saveImg(const Eigen::VectorXd &x, int rows, int cols, const std::string fname)
{
    Eigen::Array<unsigned char, -1, -1> bits = (x).cast<unsigned char>();
    cv::Mat img(rows, cols, CV_8UC1, bits.data());
    cv::imwrite(fname, img);
}

int main(int argc, char *argv[]) // EigenDMatPath, rows, cols
{
    if (argc != 4)
        return -1;
    std::string in_dir(argv[1]);
    int rows = atoi(argv[2]);
    int cols = atoi(argv[3]);

    Eigen::VectorXd u;
    egl::readDMAT(in_dir + "/u.dmat", u);
    saveImg(u, rows, cols, in_dir + "/blur.png");
    return 0;
}