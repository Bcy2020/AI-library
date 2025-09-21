#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layers.h"
#include "DataBase.h"
#include "function_namespace.h"
#include <fstream>
#include "EigenIO.h"

using namespace std;
using namespace Eigen;

class Base_Network
{
	public:
		virtual VectorXd push_forward(const VectorXd& input) = 0;
		virtual void backprop(const VectorXd& output, const VectorXd& target) = 0;
		virtual double training(int times) = 0;
		virtual double test(int times) = 0;
		virtual void model_export(const string file_name) = 0;
		virtual void model_import(const string file_name) = 0;
		virtual void learning_rate(double rate) = 0;
		~Base_Network() = default;
};

class NeuralNetwork:public Base_Network
{
	private:
		int net_size=0;
		vector<unique_ptr<Base_Layer>>layers;
		double learn_rate;
		Lossfun_pair loss_function;
		DataBase* datas;
		int batch_size;
		string ID;

	public:
		NeuralNetwork()=default;

		NeuralNetwork(NetConfig config,DataBase* Datas=NULL,Lossfun_pair lossfun=Loss_function::MSE)
			:net_size(config.layers.size()-1),ID(config.ID),learn_rate(config.learning_rate),batch_size(config.batch),datas(Datas),loss_function(lossfun)
		{
			assert(net_size>0 && "Must larger than 1 layer!");
			layers.emplace_back(make_unique<Layer>(1, config.layers[0].out_size));
			for (int i = 1; i <= net_size; i++)
			{
				if (config.layers[i].ID == "FCL")layers.emplace_back(make_unique<Layer>(config.layers[i - 1].out_size,
					config.layers[i].out_size, config.layers[i].function, config.layers[i].restriction));
			}
		}
	
		VectorXd push_forward(const VectorXd& input) override
		{
			static_cast<Layer*>(layers[0].get())->change_outs(input);
			for (int i = 1; i <= net_size; i++)
			{
				layers[i]->forward(layers[i-1]->output());
			}
			return layers[net_size]->output();
		}
		
		void backprop(const VectorXd& output,const VectorXd& target) override
		{
			VectorXd delta=loss_function.loss_derivative(output,target);
			for(int i=net_size;i>=1;i--)
			{
				layers[i]->backprop(delta,layers[i-1]->output());
				if(i!=1)delta=static_cast<Layer*>(layers[i-1].get())->get_delta(delta,layers[i]->weight());
			}
		}
		
		double training(int times) override
		{
			assert(datas!=NULL && "No DataBase!");
			double ret=0;
			random_device rd;
			mt19937 g(rd());
			for(int i=1;i<=times;i++)
			{
				Data_pair* tmp = datas->get_training_datas(g, batch_size);
				for(int j=1;j<=batch_size;j++)
				{
					tmp->input;
					VectorXd out=push_forward(tmp->input);
					backprop(out,tmp->target);
					ret+=loss_function.loss(out,tmp->target);
					tmp++;
				}
				for(int k=net_size;k>=1;k--) layers[k]->update(learn_rate);
			}
			ret/=(times*batch_size);
			return ret;
		}

		double test(int times) override
		{
			assert(datas!=NULL && "No DataBase!");
			double ret=0;
			random_device rd;
			mt19937 g(rd());
			for(int i=1;i<=times;i++)
			{
				Data_pair tmp=datas->get_testing_datas(g);
				VectorXd out=push_forward(tmp.input);
				ret+=loss_function.loss(out,tmp.target);
			}
			ret/=times;
			return ret;
		}

		void model_export(const string file_name) override
		{
			ofstream file(file_name);
			assert(file.is_open() && "Error:Cannot open or create the file");
			file<<net_size<<" "<<learn_rate<<" "<<batch_size<<endl;
			file<<loss_function.ID<<endl<<endl;
			for (int i = 1; i <= net_size; i++)
				layers[i]->layer_export(file);
		}

		void model_import(const string file_name) override
		{
			ifstream file(file_name);
			assert(file.is_open() && "Error:Cannot open or create the file");
			file>>net_size>>learn_rate>>batch_size;
			string loss_fun_ID;
			file>>loss_fun_ID;
			loss_function=lossfun_map[loss_fun_ID];
			layers.clear();
			layers.emplace_back(make_unique<Layer>(1,1));
			for(int i=1;i<=net_size;i++)
			{
				string ID;
				file >> ID;
				Base_Layer* tmp=nullptr;
				if (ID == "FCL")
					static_cast<Layer*>(tmp)->layer_import(file);
				layers.emplace_back(tmp);
			}
			layers[0]->outsize(layers[1]->insize());
    		static_cast<Layer*>(layers[0].get())->change_outs(VectorXd::Zero(layers[1]->insize()));
		}
		void database(DataBase* data){datas=data;}
		void learning_rate(double rate)override{learn_rate=rate;}
};

#endif
