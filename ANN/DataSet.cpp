#include "DataSet.h"
#include <sstream>
#include <iostream>


DataSet::DataSet(const string &filename, unsigned int n_inputs)
{
	m_n_inputs = n_inputs;
	m_data_set.open(filename.c_str());
	if (m_data_set.fail())
		cout << "chyba" << endl;
}

DataSet::~DataSet() {

}

bool DataSet::isEof(void)
{
	return m_data_set.eof();
}
void DataSet::get_min(vector<double> &input_min, vector<double> &target_min)
{
	get_next_inputs(input_min);
	get_next_targets(target_min);
}

void DataSet::get_max(vector<double> &input_max, vector<double> &target_max)
{
	get_next_inputs(input_max);
	get_next_targets(target_max);
}

void DataSet::get_next_inputs(vector<double> &input)
{
	input.clear();
	string line;
	for (unsigned int i = 0; i < m_n_inputs; i++) {
		getline(m_data_set, line, ';');
		stringstream s(line);
		double value;
		s >> value;
		input.push_back(value);
	}
}

void DataSet::get_next_targets(vector<double> &target)
{
	target.clear();
	string line;
	getline(m_data_set, line);
	stringstream s(line);
	double value;
	s >> value;
	target.push_back(value);
}


