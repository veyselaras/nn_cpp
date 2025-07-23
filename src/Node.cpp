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
	
	Node(int layerSize = 0)
	{
		std::size_t fanOut = 0;
	
		switch (layerSize)
		{
			case NNParams::INPUT:  
				fanOut = NNParams::H1;
				break;
			case NNParams::H1:     
				fanOut = NNParams::H2;
				break;
			case NNParams::H2:     
				fanOut = NNParams::OUTPUT;
				break;
			case NNParams::OUTPUT: 
				fanOut = 0;
				break;
			default:
				std::cout<<"DEFATULT DEGER"<<endl;
				break;
		}
	
		weights.resize(fanOut);
		if (fanOut > 0) 
			calculator.randomValueGenerator(weights);
	}


	void setInput(const double input, const int neuronNumber){
		this->input = input;
		if(neuronNumber == 100 || neuronNumber == 300)
			output = calculator.reluFunc(this->input);
		else if(neuronNumber == 784)
			output = input;
		
	}
	
	double getInput(){
		return input;
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