#ifndef _INTERFERENCE_MODEL_H__
#define _INTERFERENCE_MODEL_H__

#include <string>
#include <sstream>
#include <fstream>
#include <map>
#include <vector>
#include <utility>
#include <cmath>

using namespace std;

namespace interference_modeling {
	class interference_model{
	public:
		map<string, vector<double>> step1_constants;
		vector<double> step2_constants_le_100_SM;
		vector<double> step2_constants_gt_100_SM;

		map<pair<string, int>, vector<double>> step2_model_constants;

		void setup();
		void setup(string input_file1, string input_file2, string model_file);

		double get_latency(string my_model, int my_batch, int my_thread_cap, string your_model, int your_batch, int your_thread_cap);

		double get_baselatency(string model, int batch, int thread_cap);
		double get_interference(string my_model, int my_batch, string your_model, int your_batch);
	};
}
#endif
