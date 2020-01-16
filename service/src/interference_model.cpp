#include "interference_model.h"

void interference_modeling::interference_model::setup()
{
	this->setup(string("Step1_latency.CSV"), string("Step2_latency_constant.CSV"), string("Step2_latency.CSV"));
}

void interference_modeling::interference_model::setup(string input_file1, string input_file2, string model_file)
{
	//example
	/*
	 *
	 * interference_modeling::interference_model a;
	 * a.setup();
	 * cout << a.get_latency(string("vgg16"), 7, 45, string("resnet18"), 7, 55) << endl;
	 *
	 *
	 */

	//set step1_constants
	
	ifstream infile1(input_file1);
	string line;
	string token;

	getline(infile1, line);

	while(getline(infile1, line)){
		string model_name;

		istringstream ss(line);
		
		getline(ss, token, ',');
		model_name=token;

		getline(ss, token, ',');
		interference_modeling::interference_model::step1_constants[model_name].push_back(stod(token));

		getline(ss, token, ',');
		interference_modeling::interference_model::step1_constants[model_name].push_back(stod(token));
	}

	//set step2_constants

	ifstream infile2(input_file2);

	getline(infile2, line);

	getline(infile2, line);
	istringstream ss2(line);

	for(int i=0;i<5;i++){
		getline(ss2, token, ',');
		interference_modeling::interference_model::step2_constants_le_100_SM.push_back(stod(token));
	}

	getline(infile2, line);
	istringstream ss3(line);

	for(int i=0;i<5;i++){
		getline(ss3, token, ',');
		interference_modeling::interference_model::step2_constants_gt_100_SM.push_back(stod(token));
	}

	//set step2_model_constants

	ifstream infile3(model_file);

	getline(infile3, line);

	while(getline(infile3, line)){
		pair<string, int> model_batch;
		vector<double> const_set;

		istringstream ss(line);

		getline(ss, token, ',');
		model_batch.first=token;

		getline(ss, token, ',');
		model_batch.second=stoi(token);

		for(int i=0;i<4;i++){
			getline(ss, token, ',');
			const_set.push_back(stod(token));
		}

		interference_modeling::interference_model::step2_model_constants[model_batch]=const_set;
	}
}

double interference_modeling::interference_model::get_latency(string my_model, int my_batch, int my_thread_cap, string your_model, int your_batch, int your_thread_cap)
{
	double T1=interference_modeling::interference_model::get_baselatency(my_model, my_batch, my_thread_cap);
	double T2=interference_modeling::interference_model::get_baselatency(your_model, your_batch, your_thread_cap);
	double alpha=interference_modeling::interference_model::get_interference(my_model, my_batch, your_model, your_batch);
	return (T1+T2)/2.0*alpha/2.0;
}

double interference_modeling::interference_model::get_baselatency(string model, int batch, int thread_cap)
{
	return interference_modeling::interference_model::step1_constants[model][0]*((double)batch/thread_cap) + interference_modeling::interference_model::step1_constants[model][1];
}

double interference_modeling::interference_model::get_interference(string my_model, int my_batch, string your_model, int your_batch)
{
	int my_batch_below=(int)exp2((int)log2(my_batch));
	int my_batch_top=min(32, my_batch_below*2);

	int your_batch_below=(int)exp2((int)log2(your_batch));
	int your_batch_top=min(32, your_batch_below*2);

	pair<string, int> my_info_below(my_model, my_batch_below);
	pair<string, int> my_info_top(my_model, my_batch_top);

	pair<string, int> your_info_below(your_model, your_batch_below);
	pair<string, int> your_info_top(your_model, your_batch_top);

	double my_batch_ratio=1.0;
	if(my_batch_top!=my_batch_below){
		my_batch_ratio=(double)(my_batch-my_batch_below)/(my_batch_top-my_batch_below);
	}

	double your_batch_ratio=1.0;
	if(your_batch_top!=your_batch_below){
		your_batch_ratio=(double)(your_batch-your_batch_below)/(your_batch_top-your_batch_below);
	}

	double sum_theoretical_occu = (interference_modeling::interference_model::step2_model_constants[my_info_top][0]*my_batch_ratio + interference_modeling::interference_model::step2_model_constants[my_info_below][0]*(1-my_batch_ratio)) + (interference_modeling::interference_model::step2_model_constants[your_info_top][0]*your_batch_ratio+interference_modeling::interference_model::step2_model_constants[your_info_below][0]*(1-your_batch_ratio));

	double sum_avg_duration = (interference_modeling::interference_model::step2_model_constants[my_info_top][1]*my_batch_ratio + interference_modeling::interference_model::step2_model_constants[my_info_below][1]*(1-my_batch_ratio)) + (interference_modeling::interference_model::step2_model_constants[your_info_top][1]*your_batch_ratio+interference_modeling::interference_model::step2_model_constants[your_info_below][1]*(1-your_batch_ratio));

	double sum_SM_util = (interference_modeling::interference_model::step2_model_constants[my_info_top][2]*my_batch_ratio + interference_modeling::interference_model::step2_model_constants[my_info_below][2]*(1-my_batch_ratio)) + (interference_modeling::interference_model::step2_model_constants[your_info_top][2]*your_batch_ratio+interference_modeling::interference_model::step2_model_constants[your_info_below][2]*(1-your_batch_ratio));

	double sum_mem_util = (interference_modeling::interference_model::step2_model_constants[my_info_top][3]*my_batch_ratio + interference_modeling::interference_model::step2_model_constants[my_info_below][3]*(1-my_batch_ratio)) + (interference_modeling::interference_model::step2_model_constants[your_info_top][3]*your_batch_ratio+interference_modeling::interference_model::step2_model_constants[your_info_below][3]*(1-your_batch_ratio));

	if(sum_SM_util<=100.0){
		return interference_modeling::interference_model::step2_constants_le_100_SM[0]*sum_theoretical_occu+interference_modeling::interference_model::step2_constants_le_100_SM[1]*sum_avg_duration+interference_modeling::interference_model::step2_constants_le_100_SM[2]*sum_SM_util+interference_modeling::interference_model::step2_constants_le_100_SM[3]*exp(sum_mem_util)+interference_modeling::interference_model::step2_constants_le_100_SM[4];
	}
	return interference_modeling::interference_model::step2_constants_gt_100_SM[0]*sum_theoretical_occu+interference_modeling::interference_model::step2_constants_gt_100_SM[1]*sum_avg_duration+interference_modeling::interference_model::step2_constants_gt_100_SM[2]*exp(sum_SM_util)+interference_modeling::interference_model::step2_constants_gt_100_SM[3]*exp(sum_mem_util)+interference_modeling::interference_model::step2_constants_gt_100_SM[4];
}
