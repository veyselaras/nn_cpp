#include <iostream>
#include "Node.h"
#include "Calculations.h"

class Node{
public:

	double *weights;

	void setInput(const double input, bool c = 0){
		this->input = input;
		if(!c){// if c equals to 0 this node is a hiden layer neuron else may be input or output layer
			output = calculator.reluFunc(this->input);
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

private:
	
	double input;
	double output;
	double bias;
	
	Calculations calculator;
	
};