#include <cmath>
#include <vector>
#include "Calculations.h"
#include "NNParams.h"
#include "Node.h"
#include <random>
#include <cstdlib>

class Calculations{
private:
	
	std::mt19937 rng(std::random_device{}());
	
public:
	
	void randomValueGenerator(std::vector<double>& weights){
		std::size_t lengthOfWeights = weights.size();
		double sigma = std::sqrt(2.0/lengthOfWeights);
		
		for(double& x : weights) {
			double u1 = static_cast<double>(rand()) / RAND_MAX;
			double u2 = static_cast<double>(rand()) / RAND_MAX;
			double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265359 * u2);
			x = z * sigma;
		}
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

	double crossEntropy(const std::vector<double>& calculatedValues,
							  const std::vector<double>& expectedValues)
	{
		const double eps = 1e-15;
		double loss = 0.0;

		for (std::size_t i = 0; i < expectedValues.size(); ++i) {
			double p = std::clamp(calculatedValues[i], eps, 1.0 - eps);
			loss -= expectedValues[i] * std::log(p);
		}
		return loss;
	}

	void softMax(std::vector<Node>& outputLayer){
		std::size_t vectorLength = outputLayer.size();
		double sumOfInputValues = 0;
		
		for(std::size_t i = 0; i < vectorLength; i++)
			sumOfInputValues += exp(outputLayer[i].getOutput());
		
		for(std::size_t i = 0; i < vectorLength; i++)
			outputLayer[i].setOutput(exp(outputLayer[i].getOutput())/sumOfInputValues);
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