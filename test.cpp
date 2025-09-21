//#define EIGEN_USE_MKL_ALL  // �����ڰ���Eigenǰ����
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
        std::cerr << "�������!" << std::endl;
    }
    return elapsed.count();
}

int main() {
    std::cout << "=== Eigen MKL���ٲ��� ===" << std::endl;
    std::cout << "�����С: " << MATRIX_SIZE << "x" << MATRIX_SIZE << std::endl;

    // ����MKL���٣����ܰ���GPU֧�֣�ȡ����MKL�汾��
    double time_with_mkl = test_matrix_multiplication();
    std::cout << "MKL���ٺ�ʱ: " << time_with_mkl << " ��" << std::endl;

    return 0;
}