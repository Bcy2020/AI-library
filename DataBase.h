#ifndef DATABASE_H
#define DATABASE_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_OPENMP
#include <Eigen/Dense>
#include <iostream>
#include <type_traits>
#include <random>

using namespace std;
using namespace Eigen;

struct Data_pair
{
	VectorXd input,target;
};

class DataBase
{
	private:
		vector<Data_pair> Datas;
		double rate=0.8;
		int pin;

	public:
		DataBase(double rating)
			:rate(rating){}

		void push_back(const Data_pair& input){Datas.push_back(input);}
		int size(){return Datas.size();}

		void random()
		{
			random_device rd;
			mt19937 g(rd());
			shuffle(Datas.begin(),Datas.end(),g);
		}

		Data_pair* get_training_datas(std::mt19937& g,int num)
		{
			int size=static_cast<int>(floor(Datas.size()*rate));
			if (pin + num >= size)
			{
				pin = 0;
				shuffle(Datas.begin(), Datas.begin() + size - 1, g);
			}
			return Datas.data()+pin;
		}

		Data_pair get_testing_datas(std::mt19937& g)
		{
			int size=static_cast<int>(floor(Datas.size()*rate));
			uniform_int_distribution<int> dist(size,Datas.size()-1);
			return Datas[dist(g)];
		}

		void clear()
		{
			Datas.clear();
			pin = 0;
		}
};

#endif