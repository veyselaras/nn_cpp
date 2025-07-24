#include <cmath>
#include <vector>
#include "Calculations.h"
#include "NNParams.h"
#include <random>

class Calculations{
private:
	
	std::mt19937 rng(std::random_device{}());
	
	
public:
	
	randomValueGenerator(std::vector<double>& weights){
		std::size_t lengthOfWeights = weights.size();
		double sigma = std::sqrt(2.0/lengthOfWeights);
		std::normal_distribution<double> N(0.0, sigma);
		
		for(double& x : weights)
			x = N(rng);
	}

	double reluFunc(double input){
		if(input <= 0){
			return 0;
		}
		else{
			return input;
		}
	}

	double reluDerivFunc(double input){
		if(input <= 0){
			return 0;
		}
		else{
			return 1;
		}
	}

	double crossEntropy(double expectedValues, double calculatedValues){
		std::vector<double> crossEntropyValues;
		std::size_t vectorLength = expectedValues.size();
		for(std::size_t i = 0; i < vectorLength; i++){
			crossEntropyValues.push_back(-expectedValues[i] * log(calculatedValues[i] + 1e-15));
			crossEntropyValues[i] = 
		}
		return crossEntropyValues;
	}

	void softMax(std::vector<Node>& outputLayer){
		std::size_t vectorLength = inputValues.size();
		double sumOfInputValues = 0;
		
		for(std::size_t i = 0; i < vectorLength; i++)
			sumOfInputValues += exp(outputLayer[i].input);
		
		for(std::size_t i = 0; i < vectorLength; i++)
			outputLayer[i].setOutput(exp(inputValues[i])/sumOfInputValues);
	}

	void convolution(std::vector<Node>& inputLayer, std::vector<Node>& outputLayer){
		std::size_t lengthOfInput = inputLayer.size();
		std::size_t lengthOfOutput = outputLayer.size();
		
		for(std::size_t i = 0; i < lengthOfOutput; i++){
			double sum = 0;
			for(std::size_t j = 0; j < lengthOfInput; j++){
				sum += inputLayer[j].weights[i]*inputLayer[j].getOutput();
			}
			outputLayer[i].setInput(sum + outputLayer[i].bias);
		}
	}
};