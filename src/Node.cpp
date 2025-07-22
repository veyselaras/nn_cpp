#include <iostream>
#include "Node.h"
#include "Calculations.h"
#include "NNParams.h"


class Node{
private:
	
	double input;
	double output;
	double bias = 0;
	
	Calculations calculator;

public:

	std::vector<double> weights;
	
	Node(const int neuronNumber){
		switch(neuronNumber){
			case NNParams::INPUT:
				weights.resize(NNParams::H1);
				randomValueGenerator(weights);
				break;
			case NNParams::H1:
				weights.resize(NNParams::H2);
				randomValueGenerator(weights);
				break;
			case NNParams::H2:
				weights.resize(NNParams::OUTPUT);
				randomValueGenerator(weights);
				break;
			case NNParams::OUTPUT:
				break;
			default:
				std::cout<<"HATA WEIGHTS OLUSAMADI"<<endl;
				break;
		}
	}	
	
	void setInput(const double input, const int neuronNumber){
		this->input = input;
		if(neuronNumber == 100 || neuronNumber == 300){
			output = calculator.reluFunc(this->input);
			
		}
	}
	
	double getInput(){
		return input;
	}
	
	void setOutput(double output){
		this->output = output;
	}
	
	double getOutput(){
		return output;
	}
	
	void setBias(double output){
		this->output = output;
	}
	
	double getBias(){
		return output;
	}
};