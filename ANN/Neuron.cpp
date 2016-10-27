#include "Neuron.h"

Neuron::Neuron(unsigned int n_links, unsigned int layer_index, double learning_rate, double momentum)
{
	for (unsigned int i = 0; i < n_links; i++) {
		m_links.push_back(Link());
		m_links.back().weight = init_weight_val() / 100.0;
		m_links.back().delta_weight = 0.0;
	}

	m_layer_index = layer_index;
	m_learning_rate = learning_rate;
	m_momentum = momentum;
}

Neuron::~Neuron() {
}

void Neuron::compute_output(const Layer &prev_layer) {
	double sum = 0.0;
	unsigned int i;
	for (i = 0; i < prev_layer.size(); i++)
	{
		sum += prev_layer[i].get_output() * prev_layer[i].m_links[m_layer_index].weight;
	}
	m_output = activation_f(sum);

}

void Neuron::update_links_weight(Layer &prev_layer)
{
	unsigned int i;
	for (i = 0; i < prev_layer.size(); i++) {
		Neuron &neuron = prev_layer[i];
		double delta_weight = neuron.m_links[m_layer_index].delta_weight;
		double new_delta_weight = m_learning_rate * neuron.get_output() * m_gradient + m_momentum * delta_weight;

		neuron.m_links[m_layer_index].delta_weight = new_delta_weight;
		neuron.m_links[m_layer_index].weight += new_delta_weight;
	}
}

// zosumuje chyby neuronov v nasledujucej vrstve vynasobene vahou prepojenia
double Neuron::sum_dow(const Layer &next_layer) const
{
	double sum = 0.0;
	unsigned int i;
	for (i = 0; i < next_layer.size() - 1; i++) {
		sum += m_links[i].weight * next_layer[i].m_gradient;
	}
	return sum;
}

void Neuron::compute_hidden_gradient(const Layer &next_layer)
{
	double dow = sum_dow(next_layer);
	m_gradient = dow * activation_f_derivation(m_output);
}

void Neuron::compute_output_gradient(double target)
{
	double delta = target - m_output;
	m_gradient = delta * activation_f_derivation(m_output);
}

void Neuron::set_output(double output)
{
	m_output = output;
}

double Neuron::get_output() const
{
	return m_output;
}

double Neuron::init_weight_val(void)
{
	return rand() / double(RAND_MAX);
}

double Neuron::activation_f(double x)
{
	return tanh(x);
	// return 1 / (1 + exp(-x));
}

double Neuron::activation_f_derivation(double x)
{
	return 1.0 - tanh(x) * tanh(x);
	// return activation_f(x) * (1 - activation_f(x));
}