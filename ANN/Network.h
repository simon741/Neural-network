#pragma once

#include <vector>
#include "Neuron.h"

using namespace std;


typedef struct {
	vector<unsigned int> layers_structure;
	double learning_rate; //rychlost ucenia
	double momentum;       //zabranuje upadnutiu do lokalneho minima
} NetConf;

class Network
{
public:
	Network(NetConf &net_conf);
	~Network();

	void feed_forward(const vector<double> &input);
	double get_last_error(const vector<double> &target);
	void back_prop(const vector<double> &target);
	void get_result(vector<double> &results);

private:
	vector<Layer> m_layers;
	void init_input_layer(const vector<double> &input);
};

