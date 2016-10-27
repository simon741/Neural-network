#pragma once

#include <vector>
#include <string>
#include <fstream>

using namespace std;

class DataSet
{
public:
	DataSet(const std::string &filename, unsigned int dataset_size);
	~DataSet();

	bool isEof(void);
	void get_min(vector<double> &input_min, vector<double> &target_min);
	void get_max(vector<double> &input_max, vector<double> &target_max);
	void get_next_inputs(vector<double> &input);
	void get_next_targets(vector<double> &target);

private:
	ifstream m_data_set;
	unsigned int m_n_inputs;
};



