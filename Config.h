#include<iostream>
#include"function_namespace.h"
#include<vector>
#include <nlohmann/json.hpp>
#include <fstream>
using namespace std;
using json = nlohmann::json;

struct LayerConfig
{
    string ID;
    int out_size;
    Actfun_pair function;
    grad_restriction_function restriction = grad_restriction::empty;
};
typedef vector<LayerConfig> LayerConfig_Vector;

LayerConfig_Vector build_layers(int input_size, int output_size, int net_size, int layer_size)
{
    assert(net_size > 0 && "Must have at least 1 hidden layer!");
    LayerConfig_Vector config(net_size + 1);
    config[0].out_size = input_size;
    config[0].function = Activition::Sigmoid;
    config[0].ID = "FCL";
    for (int i = 1; i < net_size; i++)
    {
        config[i].ID = "FCL";
        config[i].out_size = layer_size;
        config[i].function = Activition::ReLU;
    }
    config[net_size].out_size = output_size;
    config[net_size].function = Activition::Sigmoid;
    config[net_size].ID = "FCL";
    return config;
}

struct NetConfig
{
    string ID;
    double learning_rate;
    LayerConfig_Vector layers;
    int batch;
};

NetConfig build_net(string ID, double learning_rate, const LayerConfig_Vector& layers)
{
    return{ ID,learning_rate,layers };
}

void read_layeronfig_from_json(string file_name, NetConfig& config)
{
    LayerConfig_Vector layers_config;
    layers_config.clear();
    json config_from_file;
    ifstream file(file_name);
    if (file.is_open())
    {
        file >> config_from_file;
        file.close();
    }
    else assert("Cannot Open The File!");
    if (!(config_from_file.contains("input_size") && config_from_file.contains("output_size") && config_from_file.contains("learning_rate") && config_from_file.contains("layers")
         && config_from_file.contains("batch")))assert("Illegal config json!");
    int input_size = config_from_file["input_size"];
    int output_size = config_from_file["output_size"];
    config.ID = "empty";
    if (config_from_file.contains("ID"))config.ID = config_from_file["ID"];
    config.batch = config_from_file["batch"];
    config.learning_rate = config_from_file["learning_rate"];
    if (!config_from_file["layers"].is_array())assert("Illegal layer Configs!");
    layers_config.push_back({ "FCL",input_size,actfun_map["ReLU"],grad_restriction::empty });
    for (auto& layer : config_from_file["layers"])
    {
        LayerConfig tmp;
        if (!(layer.contains("output_size") && layer.contains("ID") && layer.contains("activition")))assert("AN Illegal layer!");
        tmp.ID = layer["ID"];
        tmp.out_size = layer["output_size"];
        tmp.function = actfun_map[layer["activition"]];
        if (layer["restriction"] == "empty" || layer.contains("restriction") == false)tmp.restriction = grad_restriction::empty;
        else if (layer["restriction"] == "cropping")
        {
            if (!layer.contains("cropping_rate"))assert("Lack of cropping rate!");
            else tmp.restriction = grad_restriction::make_cropping_function(layer["restriction"]);
        }
        else assert("Unknown restriction function!");
        layers_config.push_back(tmp);
    }
    config.layers = layers_config;
}