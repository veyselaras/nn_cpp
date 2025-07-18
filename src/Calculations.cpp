#include <cmath>
#include <vector>
#include "Calculations.h"

class Calculations{
public:
	
	double reluFunctios(double input){
		if(input <= 0){
			return 0;
		}
		else{
			return input;
		}
	}
	
	std::vector<double> crossEntropy(const std::vector<double>& expectedValues, const std::vector<double>& calculatedValues){
		std::vector<double> crossEntropyValues;
		int vectorLength = expectedValues.size();
		for(int i = 0; i < vectorLength; i++)
			crossEntropyValues.push_back(-expectedValues[i] * log(calculatedValues[i] + 1e-15));
		
		return crossEntropyValues;
	}
	
	void softMax(const std::vector<double>& inputValues, std::vector<Node>& outputLayer){
		int vectorLength = inputValues.size();
		double sumOfInputValues = 0;
		
		for(int i = 0; i < vectorLength; i++)
			sumOfInputValues += exp(inputValues[i]);
		
		for(int i = 0; i < vectorLength; i++)
			outputLayer[i].setOutput(exp(inputValues[i])/sumOfInputValues);
	}
	
	void convolution(){
		
	}
	
	void updateFuction(){
		
	}
	
	void delta(){
		
	}
	
	void gradient(){
		
	}
	
private:
	
	
	
};