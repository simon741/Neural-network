#include <vector>
#include <iostream>
#include <cassert>
#include <sstream>
#include <omp.h>
#include "Network.h"
#include "DataSet.h"
#define N_INPUTS 4
#define N_OUTPUTS 1

using namespace std;

void create_hidden_layer_structure(unsigned int n_layers, unsigned int neurons_in_layer, vector<unsigned int> &layers_structure)
{
	for (unsigned int i = 0; i < n_layers; i++) {
		layers_structure.push_back(neurons_in_layer);
	}
}

void print_vector(string label, vector<double> &v, ofstream &stream)
{
	string s;
	stream << label;
	for (unsigned int i = 0; i < v.size(); i++) {
		stream << v[i] << " ";
	}
	stream << endl;
}

//scale into [-1,1]
void normalize(const vector<double> &min, const vector<double> &max, vector<double> &x)
{
	for (unsigned int i = 0; i < x.size(); i++)
		x[i] = 2 * (x[i] - min[i]) / (max[i] - min[i]) - 1;
}

void denormalize(const vector<double> &min, const vector<double> &max, vector<double> &x)
{
	for (unsigned int i = 0; i < x.size(); i++)
		x[i] = (x[i] + 1) * (max[i] - min[i]) / 2 + min[i];
}

void train_net(Network &net, DataSet &data_set, vector<double> &input_min, vector<double> &input_max,
	vector<double> &target_min, vector<double> &target_max, NetConf &net_conf)
{
	vector<double> input, target, output;
	double recent_average_e = 0.0;

	stringstream filename;
	filename << "train_" << net_conf.layers_structure.size() - 2 << "x" << net_conf.layers_structure[1] << \
		"_lr(" << net_conf.learning_rate << ")_m(" << net_conf.momentum << ").txt";
	const string s(filename.str());
	ofstream train_output;
	train_output.open(s.c_str(), std::ofstream::out | std::ofstream::trunc);
	if (train_output.fail())
		cout << "chyba" << endl;

	train_output << "---------------Vysledky Trenovania-------------" << endl;
	train_output << "error is frominterval <0,2>" << endl;
	unsigned int epoch = 0;
	while (epoch <= 1500 && !data_set.isEof()) { //vypis do suboru
		epoch++;
		train_output << endl << "Epoch: " << epoch << endl;

		data_set.get_next_inputs(input);
		print_vector("Inputs: ", input, train_output);
		normalize(input_min, input_max, input);

		net.feed_forward(input);

		net.get_result(output);
		denormalize(target_min, target_max, output);
		print_vector("Outputs: ", output, train_output);

		data_set.get_next_targets(target);
		print_vector("Targets: ", target, train_output);
		normalize(target_min, target_max, target);

		net.back_prop(target);

		//priemerna RMSE poslednych 100 iteracii
		if (epoch < 100) {
			recent_average_e = (recent_average_e * (epoch - 1) + net.get_last_error(target)) / epoch;
		}
		else {
			recent_average_e = (recent_average_e * 100.0 + net.get_last_error(target)) / 101.0;
			train_output << "Training recent average error(last 100 epochs): " << recent_average_e << endl;
		}
	}
	train_output.close();
}

double test_net(Network &net, DataSet &data_set, vector<double> &input_min, vector<double> &input_max,
	vector<double> &target_min, vector<double> &target_max, NetConf &net_conf)
{
	vector<double> input, target, output;

	stringstream filename;
	filename << "test_" << net_conf.layers_structure.size() - 2 << "x" << net_conf.layers_structure[1] << \
		"_lr(" << net_conf.learning_rate << ")_m(" << net_conf.momentum << ").txt";
	const string s(filename.str());
	ofstream test_output;
	test_output.open(s.c_str(), std::ofstream::out | std::ofstream::trunc);
	if (test_output.fail())
		cout << "chyba" << endl;

	double average_e = 0.0;
	unsigned int test_n = 0;

	test_output << "---------------Vysledky Testovania-------------" << endl;
	test_output << "error is frominterval <0,2>" << endl;
	while (test_n < 100 && !data_set.isEof()) {
		test_n++;
		test_output << endl << "test number: " << test_n << endl;
		data_set.get_next_inputs(input);
		print_vector("Inputs: ", input, test_output);
		normalize(input_min, input_max, input);

		net.feed_forward(input);

		data_set.get_next_targets(target);
		print_vector("Targets: ", target, test_output);

		net.get_result(output);
		denormalize(target_min, target_max, output);
		print_vector("Outputs: ", output, test_output);

		normalize(target_min, target_max, target);
		average_e = ((test_n - 1) * average_e + net.get_last_error(target)) / test_n;
		test_output << "Testing average error: " << average_e << endl;
	}
	test_output.close();
	return average_e;
}

int main()
{
	DataSet data_set("data.csv", N_INPUTS);

	vector<double> input_min, target_min, input_max, target_max;
	data_set.get_min(input_min, target_min);
	data_set.get_max(input_max, target_max);

	vector<NetConf> net_confs;

	cout << "Zadaj kolko roznych NS chces testovat" << endl;
	unsigned n;
	cin >> n;
	for (unsigned int i = 1; i <= n; i++) {
		NetConf net_conf;
		net_conf.layers_structure.push_back(N_INPUTS);  //vstupna vrstva
		cout << "Struktura " << i << ". NS:" << endl;
		cout << "Pocet skrytych vrstiev: ";
		unsigned int n_layers;
		cin >> n_layers;
		cout << "Pocet neuronov v skrytej vrstve: ";
		unsigned int neurons_in_layer;
		cin >> neurons_in_layer;
		create_hidden_layer_structure(n_layers, neurons_in_layer, net_conf.layers_structure); //skryta vrstva
		net_conf.layers_structure.push_back(N_OUTPUTS);  //vystupna vrstva
		cout << "Rychlost ucenia: ";
		cin >> net_conf.learning_rate;
		cout << "Momentum: ";
		cin >> net_conf.momentum;
		net_confs.push_back(net_conf);
	}

	double min_test_err;
	double test_err;
	NetConf *best_net_conf;
	double start_time = omp_get_wtime();
	for (unsigned int i = 0; i < n; i++) {

		Network net(net_confs[i]);

		train_net(net, data_set, input_min, input_max, target_min, target_max, net_confs[i]);

		test_err = test_net(net, data_set, input_min, input_max, target_min, target_max, net_confs[i]);
		if (i == 0 || test_err < min_test_err) {
			min_test_err = test_err;
			best_net_conf = &net_confs[i];
		}
	}
	double stop_time = omp_get_wtime();
	cout << endl << "---------------Vysledok-------------" << endl;
	cout << "Zo zadanych konfiguracii neuronovej siete ma najlepsie vysledky  konfiguracia s parametrami: " << endl;
	cout << "Skrytych vrstiev: " << (best_net_conf->layers_structure).size() - 2 << endl;
	cout << "Neuronov v skrytej vrstve: " << best_net_conf->layers_structure[1] << endl;
	cout << "Rychlost ucenia: " << best_net_conf->learning_rate << endl;
	cout << "Momentum: " << best_net_conf->momentum << endl;
	cout << "Priemerna chyba na testovacej mnozine je: " << min_test_err * 50 << "%" << endl;
	cout << "Cas vykonavania: " << stop_time - start_time << " s" << endl;
}
