//#define EIGEN_USE_MKL_ALL  // 必须在包含Eigen前定义
#include <iostream>
#include <chrono>
#include <Eigen/Dense>

const int MATRIX_SIZE = 2000;

double test_matrix_multiplication() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::MatrixXd B = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::MatrixXd C(MATRIX_SIZE, MATRIX_SIZE);

    auto start = std::chrono::high_resolution_clock::now();
    C = A * B;
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    if (std::isnan(C(0, 0))) {
        std::cerr << "计算错误!" << std::endl;
    }
    return elapsed.count();
}

int main() {
    std::cout << "=== Eigen MKL加速测试 ===" << std::endl;
    std::cout << "矩阵大小: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;

    // 测试MKL加速（可能包含GPU支持，取决于MKL版本）
    double time_with_mkl = test_matrix_multiplication();
    std::cout << "MKL加速耗时: " << time_with_mkl << " 秒" << std::endl;

    return 0;
}