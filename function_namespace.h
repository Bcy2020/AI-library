#ifndef FUNCTION_NAMESPACE_H
#define FUNCTION_NAMESPACE_H

#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_OPENMP
#include <Eigen/Dense>
#include <iostream>
#include <map>

using namespace std;
using namespace Eigen;

using Activefun=function<VectorXd(const VectorXd&)>;
struct Actfun_pair
{
	Activefun act,act_der;
	string ID;
};

map<string,Actfun_pair>actfun_map;
namespace Activition
{
	inline VectorXd linear(const VectorXd& inp){return inp;}
	inline VectorXd linear_derivative(const VectorXd& inp){return VectorXd::Ones(inp.size());}
	const Actfun_pair Linear={linear,linear_derivative,"Linear"};

	inline VectorXd sigmoid(const VectorXd& inp)
		{return 1.0/(1.0+(-inp).array().exp());}
	inline VectorXd sigmoid_derivative(const VectorXd& inp)
	{
		auto temp=sigmoid(inp);
		return temp.array()*(1.0-temp.array());
	}
	const Actfun_pair Sigmoid={sigmoid,sigmoid_derivative,"Sigmoid"};

	inline VectorXd relu(const VectorXd& inp)
		{return inp.array().max(0.0);}
	inline VectorXd relu_derivative(const VectorXd& inp)
		{return (inp.array()>0.0).cast<double>(); }
	const Actfun_pair ReLU={relu,relu_derivative,"ReLU"};

	inline VectorXd softmax(const VectorXd& inp)
	{
		VectorXd exp = (inp.array() - inp.maxCoeff()).exp();
		return exp / exp.sum();
	}
	inline VectorXd softmax_derivative(const VectorXd& inp)
	{
		VectorXd tmp = softmax(inp);
		MatrixXd ret = tmp.asDiagonal().toDenseMatrix() - tmp * tmp.transpose();
		return ret.diagonal();
	}
	const Actfun_pair Softmax = { softmax,softmax_derivative,"Softmax" };
}

namespace
{
	struct ActfunInitializer
	{
		ActfunInitializer()
		{
			actfun_map["Sigmoid"]=Activition::Sigmoid;
			actfun_map["ReLU"]=Activition::ReLU;
			actfun_map["Linear"]=Activition::Linear;
			actfun_map["Softmax"] = Activition::Softmax;
		}
	};
	ActfunInitializer Actfun_Initializer;
}

using Lossfun=function<double(const VectorXd&,const VectorXd&)>;
using Lossfun_der=function<VectorXd(const VectorXd&,const VectorXd&)>;
struct Lossfun_pair
{
	Lossfun loss;
	Lossfun_der loss_derivative;
	string ID;
};
map<string,Lossfun_pair> lossfun_map;
namespace Loss_function
{
	inline double mse(const VectorXd& output,const VectorXd& target)
	{
		VectorXd diff=output-target;
		return diff.squaredNorm()/output.size();
	}
	inline VectorXd mse_derivative(const VectorXd& output,const VectorXd& target)
	{
		return 2*(output-target)/output.size();
	}
	Lossfun_pair MSE={mse,mse_derivative,"MSE"};
	
    inline double cross_entropy(const VectorXd& output, const VectorXd& target)
	{
	    const double eps = 1e-10;
	    return -(target.array()*(output.array()+eps).log()+ 
	            (1-target.array())*(1-output.array()+eps).log()).mean();
    }
	inline VectorXd cross_entropy_derivative(const VectorXd& output, const VectorXd& target)
	{
		return output - target;
	}
	Lossfun_pair CRE = { cross_entropy,cross_entropy_derivative,"CRE" };
}
namespace
{
	struct LossfunInitializer
	{
		LossfunInitializer()
		{
			lossfun_map["MSE"] = Loss_function::MSE;
			lossfun_map["CRE"] = Loss_function::CRE;
		}
	};
	LossfunInitializer Lossfun_Initializer;
}

using grad_restriction_function = function<void(VectorXd&)>;
namespace grad_restriction
{
	inline void empty(VectorXd& delta) {}

	inline void cropping(VectorXd& delta, double threshold)
	{
		double norm = delta.norm();
		if (norm > threshold)delta *= (threshold / norm);
	}

	inline grad_restriction_function make_cropping_function(double threshold)
	{
		return [threshold](VectorXd& delta) {cropping(delta, threshold); };
	}
}

#endif