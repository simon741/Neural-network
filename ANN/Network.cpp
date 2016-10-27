#include "Network.h"

Network::Network(NetConf &net_conf)
{
	vector<unsigned int> &layers_structure = net_conf.layers_structure;
	for (unsigned int i = 0; i < layers_structure.size(); i++) {
		m_layers.push_back(Layer());
		unsigned int n_links = 0;
		if (i < layers_structure.size() - 1) {
			n_links = layers_structure[i + 1];
		}
		for (unsigned int j = 0; j < layers_structure[i]; j++) {
			m_layers.back().push_back(Neuron(n_links, j, net_conf.learning_rate, net_conf.momentum));
		}
		Neuron bias(n_links, m_layers.size(), net_conf.learning_rate, net_conf.momentum);
		bias.set_output(1.0);
		m_layers.back().push_back(bias);
	}
}

Network::~Network() {

}

void Network::feed_forward(const vector<double> &input)
{
	init_input_layer(input);
	unsigned int j;
	unsigned int i;

#pragma omp parallel default(shared) private(i,j)
	for (i = 1; i < m_layers.size(); i++) {
		Layer &layer = m_layers[i];
		Layer &prev_layer = m_layers[i - 1];
#pragma omp for
		for (j = 0; j < m_layers[i].size() - 1; j++) { //bez biasu
			layer[j].compute_output(prev_layer);
		}
	}
}

double Network::get_last_error(const vector<double> &target)
{
	Layer &output_layer = m_layers.back();
	double error = 0.0;
	for (unsigned int i = 0; i < output_layer.size() - 1; i++) {
		double x = target[i] - output_layer[i].get_output();
		error += x * x;
	}
	error /= output_layer.size() - 1;
	error = sqrt(error); //RMSE
	return error;
}

void Network::back_prop(const vector<double> &target)
{
	Layer &output_layer = m_layers.back();


	//vypocet gradientu pre vystupne neurony
	for (unsigned int i = 0; i < output_layer.size() - 1; i++) {
		output_layer[i].compute_output_gradient(target[i]);
	}

	//vypocet gradientu pre skryte vrstvy neuronov
	unsigned int j;
	unsigned int i;
#pragma omp parallel default(shared) private(i,j)
	for (i = m_layers.size() - 2; i> 0; i--) { //od poslednej skrytej po prvu skrytu
		Layer &layer = m_layers[i];
		Layer &next_layer = m_layers[i + 1];
#pragma omp for
		for (j = 0; j < layer.size(); j++) {
			layer[j].compute_hidden_gradient(next_layer);
		}
	}

	//aktualizacia vah prepojeni medzi vrstvami
#pragma omp parallel default(shared) private(i,j)
	for (i = m_layers.size() - 1; i > 0; --i) { //od vystupnej po prvu skrytu
		Layer &layer = m_layers[i];
		Layer &prev_layer = m_layers[i - 1];
#pragma omp for
		for (j = 0; j < layer.size() - 1; j++) { //layer.size() - 1 lebo bias nema ziadne prepojenie z predoslou vrstvou
			layer[j].update_links_weight(prev_layer);
		}
	}
}

void Network::get_result(vector<double> &results)
{
	results.clear();
	Layer &output_layer = m_layers.back();

	for (unsigned int i = 0; i < output_layer.size() - 1; i++) {
		results.push_back(output_layer[i].get_output());
	}
}

void Network::init_input_layer(const vector<double> &input) {
	for (unsigned int i = 0; i < input.size(); i++) {
		m_layers[0][i].set_output(input[i]);
	}
}
