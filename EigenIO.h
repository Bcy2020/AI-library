#ifndef EIGENIO_H
#define EIGENIO_H
#include<Eigen/Dense>
#include<fstream>

using namespace std;
using namespace Eigen;

MatrixXd Eigen_read_from_file(ifstream& file, int rows, int cols)
{
	MatrixXd ret(rows, cols);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			file >> ret(i, j);
		}
	return ret;
}

#endif