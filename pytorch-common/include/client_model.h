#ifndef CLIENT_MODEL_H__
#define CLIENT_MODEL_H__

using namespace std;

class client_model
{

public:
client_model(int nclient, double lambda);
~client_model();

double getRandNumber();
double randomExponentialInterval(double mean, unsigned int seed);

private:
int _nclient;
double _lambda;
};


#else
#endif
