#ifndef LAYERS_H
#define LAYERS_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_OPENMP
#include <Eigen/Dense>
#include <iostream>
#include <type_traits>
#include <random>
#include <map>
#include "function_namespace.h"
#include "EigenIO.h"
#include "Config.h"

using namespace std;
using namespace Eigen;

/*template <typename eig>
void random_init(MatrixBase<eig>& mat,typename eig::Scalar Min=-1.0,typename eig::Scalar Max=1.0)
{
	static_assert(!is_const_v<eig>,"Input illigal");
	using sca=eig::Scalar;
	random_device rd;
	mt19937 rnd(rd());
	
	if constexpr (is_integral_v<sca>)
	{
		uniform_int_distribution<sca> dist(Min,Max);
		for(int i=0;i<mat.size();i++) mat(i)=dist(rnd);
	}
	else
	{
		uniform_real_distribution<sca> dist(Min,Max);
		for(int i=0;i<mat.size();i++) mat(i)=dist(rnd);
	}
}*/


class Base_Layer
{
	protected:
		int in_size,out_size;
		grad_restriction_function grad_restriction_fun;
		Actfun_pair act_fun;

	public:
		virtual VectorXd forward(const VectorXd& input)=0;
		virtual void backprop(const VectorXd& delta,const VectorXd& front_input)=0;
		virtual void update(double learn_rate)=0;
		virtual void grad_init()=0;
		virtual void grad_restriction_fun_change(grad_restriction_function func)=0;
		
		virtual void insize(int size)=0;
		virtual int insize()=0;

		virtual void outsize(int size)=0;
		virtual int outsize()=0;

		virtual string actfun()=0;
		virtual void actfun(const Actfun_pair& act)=0;

		virtual MatrixXd weight()=0;
		virtual void weight(const MatrixXd& tmp)=0;

		virtual MatrixXd biase()=0;
		virtual void biase(const MatrixXd& tmp)=0;

		virtual VectorXd output()=0;
		
		virtual void layer_export(ofstream& file) = 0;
		virtual void layer_import(ifstream& file) = 0;

		~Base_Layer()=default;
};

class Layer : public Base_Layer
{
	private:
		VectorXd z,act;
		MatrixXd weights_grad,biases_grad;
		grad_restriction_function grad_restriction_fun;
		MatrixXd weights,biases;
		Actfun_pair act_fun;

		friend class NeuralNetwork;
	public:
		Layer()=default;

		Layer(int ins,int outs,Actfun_pair actfp=Activition::Sigmoid,
			grad_restriction_function restriction_fun=grad_restriction::empty)
			:act_fun(actfp),grad_restriction_fun(restriction_fun)
		{
			in_size=ins;
			out_size=outs;
			double scale = sqrt(2.0 / in_size);
			MatrixXd tmp=MatrixXd::Random(outs,ins);
			weights=scale*tmp;
			biases=VectorXd::Zero(outs);
			weights_grad=MatrixXd::Zero(outs,ins);
			biases_grad=MatrixXd::Zero(outs,1);
		}
		
		VectorXd forward(const VectorXd& input) override
		{
			assert(input.size()==in_size && "Illegal input!");
			z=weights*input+biases;
			act=act_fun.act(z);
			return act;
		}
		
		VectorXd get_delta(const VectorXd& delta,const MatrixXd& next_weight)
		{
			VectorXd ret=next_weight.transpose()*delta,tmp=act_fun.act_der(z);
			ret=ret.cwiseProduct(tmp);
			grad_restriction_fun(ret);
			return ret;
		}
		
		void backprop(const VectorXd& delta,const VectorXd& front_input) override
		{
			weights_grad+=delta*front_input.transpose();
			biases_grad+=delta;
		}

		void grad_init() override
		{
			weights_grad=MatrixXd::Zero(out_size,in_size);
			biases_grad=MatrixXd::Zero(out_size,1);
		}
		
		void update(double learn_rate) override
		{
			weights-=weights_grad*learn_rate;
			biases-=biases_grad*learn_rate;
			grad_init();
		}

		void layer_export(ofstream& file) override
		{
			file << "FCL\n" << in_size << " " << out_size<<"\n";
			file << act_fun.ID << "\n";
			file << weights.rows() << " " << weights.cols() << "\n" << weights << "\n";
			file << biases.rows() << " " << biases.cols() << "\n" << biases << "\n";
			file << "\n";
		}

		void layer_import(ifstream& file) override
		{
			file >> in_size >> out_size;
			string ID;
			file >> ID;
			act_fun = actfun_map[ID];
			int row, col;
			file >> row >> col;
			weights = MatrixXd::Zero(row,col);
			weights = Eigen_read_from_file(file, row, col);
			file >> row >> col;
			biases= MatrixXd::Zero(out_size, 1);
			biases = Eigen_read_from_file(file, row, col);
			grad_init();
		}

		void grad_restriction_fun_change(grad_restriction_function func)override{grad_restriction_fun=func;}
		
		void insize(int size)override{in_size=size;}
		int insize()override{return in_size;}

		void outsize(int size)override{out_size=size;}
		int outsize()override{return out_size;}

		string actfun()override{return act_fun.ID;}
		void actfun(const Actfun_pair& act)override{act_fun=act;}

		MatrixXd weight()override{return weights;}
		void weight(const MatrixXd& tmp)override{weights=tmp;}

		MatrixXd biase()override{return biases;}
		void biase(const MatrixXd& tmp)override{biases=tmp;}

		VectorXd output()override{return act;}
		void change_outs(const VectorXd& input){act=input;}
};

#endif