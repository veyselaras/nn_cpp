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

	void BackProp(std::vector<Node>& inputLayer, 
					  std::vector<Node>& hidden1, 
					  std::vector<Node>& hidden2, 
					  std::vector<Node>& outputLayer, 
					  const double expectedValue[NNParams::OUTPUT]){
		
		double deltaOfOut[NNParams::OUTPUT];
		double deltaOfH2[NNParams::H2];
		double deltaOfH1[NNParams::H1];
		
		for(int i = 0; i < NNParams::OUTPUT; i++)
			deltaOfOut[i] = outputLayer[i].getOutput() - expectedValue[i];
		
		for(int j = 0; j < NNParams::H2; j++){
			double sum = 0;
			for(int i = 0; i < NNParams::OUTPUT; i++){
				sum += deltaOfOut[i] * hidden2[j].weights[i];
			}
			deltaOfH2[j] = sum * reluDerivFunc(hidden2[j].getOutput());
		}
		
		for(int j = 0; j < NNParams::H1; j++){
			double sum = 0;
			for(int k = 0; k < NNParams::H2; k++){
				sum += deltaOfH2[k] * hidden1[j].weights[k];
			}
			deltaOfH1[j] = sum * reluDerivFunc(hidden1[j].getOutput());
		}
		
		
		//update bias and weights
		for (int i = 0; i < NNParams::OUTPUT; ++i) {
			outputLayer[i].bias -= NNParams::LEARNINGRATE * deltaOfOut[i];
			for (int j = 0; j < NNParams::H2; ++j) {                      
				double gradient = deltaOfOut[i] * hidden2[j].out;
				hidden2[j].weights[i] -= NNParams::LEARNINGRATE * gradient;
			}
		}
		
		for (int k = 0; k < NNParams::H2; ++k) {//bias problems
			hidden2[k].bias -= NNParams::LEARNINGRATE * deltaOfH2[k];
			for (int j = 0; j < NNParams::H1; ++j) {
				double gradient = deltaOfH2[k] * hidden1[j].out;
				hidden1[j].weights[k] -= NNParams::LEARNINGRATE * gradient;
			}
		}
		
		for (int h = 0; h < NNParams::H1; ++h) {
			hidden1[h].bias -= NNParams::LEARNINGRATE * deltaOfH1[h];
			for (int k = 0; k < NNParams::INPUT; ++k) {
				double gradient = deltaOfH1[h] * inputLayer[k].out;
				inputLayer[k].weights[h] -= NNParams::LEARNINGRATE * gradient;
			}
		}
	}
};