#pragma once

#include <vector>

using namespace std;

typedef struct {
	double weight;
	double delta_weight;
} Link;

class Neuron;
typedef vector<Neuron> Layer;

class Neuron
{
public:
	Neuron(unsigned int n_links, unsigned int layer_index, double learning_rate, double momentum);
	~Neuron();

	void compute_output(const Layer &prev_layer);
	void update_links_weight(Layer &prev_layer);
	double sum_dow(const Layer &next_layer) const;
	void compute_hidden_gradient(const Layer &next_layer);
	void compute_output_gradient(double target);
	void set_output(double output);
	double get_output() const;

private:
	vector<Link> m_links;
	unsigned int m_layer_index;
	double m_output;
	double m_gradient;
	double m_learning_rate;
	double m_momentum;

	double init_weight_val(void);
	double activation_f(double x);
	double activation_f_derivation(double x);
};

