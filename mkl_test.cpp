#include <iostream>
#include "mkl.h"  // MKL头文件

int main() {
    // 测试矩阵-向量乘法 (dgemv)
    const int m = 2, n = 2;  // 矩阵大小 2x2
    const double alpha = 1.0, beta = 0.0;
    const int lda = m, incx = 1, incy = 1;
    
    // 定义矩阵A和向量x、y
    double A[4] = {1.0, 2.0, 3.0, 4.0};  // 2x2矩阵
    double x[2] = {1.0, 1.0};            // 输入向量
    double y[2];                         // 输出向量
    
    // 调用MKL的dgemv函数 (矩阵-向量乘法)
    dgemv("N", &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
    
    // 输出结果 (预期: y = [3.0, 7.0])
    std::cout << "MKL测试结果:\n";
    std::cout << "y[0] = " << y[0] << "\ny[1] = " << y[1] << std::endl;
    
    // 验证结果是否正确
    if (y[0] == 3.0 && y[1] == 7.0) {
        std::cout << "MKL安装成功，函数调用正常！" << std::endl;
        return 0;
    }
}