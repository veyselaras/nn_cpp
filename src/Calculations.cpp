#include <cmath>
#include <vector>
#include "Calculations.h"

#define learningRate 0.01

class Calculations{
public:
	
	double reluFunctions(double input){
		if(input <= 0){
			return 0;
		}
		else{
			return input;
		}
	}
	
	double reluDerivFunctions(double input){
		if(input <= 0){
			return 0;
		}
		else{
			return 1;
		}
	}
	
	double crossEntropy(double expectedValues, double calculatedValues){
		std::vector<double> crossEntropyValues;
		int vectorLength = expectedValues.size();
		for(int i = 0; i < vectorLength; i++){
			crossEntropyValues.push_back(-expectedValues[i] * log(calculatedValues[i] + 1e-15));
			crossEntropyValues[i] = 
		}
		return crossEntropyValues;
	}
	
	void softMax(std::vector<Node>& outputLayer){
		int vectorLength = inputValues.size();
		double sumOfInputValues = 0;
		
		for(int i = 0; i < vectorLength; i++)
			sumOfInputValues += exp(outputLayer[i].input);
		
		for(int i = 0; i < vectorLength; i++)
			outputLayer[i].setOutput(exp(inputValues[i])/sumOfInputValues);
	}
	
	void convolution(std::vector<Node>& inputLayer, std::vector<Node>& outputLayer){
		int lengthOfInput = inputLayer.size();
		int lengthOfOutput = outputLayer.size();
		
		double convolutionOfNode = 0;
		
		for(int i = 0; i < lengthOfOutput; i++){
			for(int j = 0; j < lengthOfInput; j++){
				convolutionOfNode += inputLayer[j].weights[i];
			}
			outputLayer[i] = convolutionOfNode + outputLayer[i].bias;
			convolutionOfNode = 0;
		}
	}
	
	void BackProp(std::vector<Node>& inputLayer, std::vector<Node>& hidden1, std::vector<Node>& hidden2, std::vector<Node>& outputLayer, double expectedValue[10]){
		
		double deltaOfOut[10];
		double deltaOfH2[100];
		double deltaOfH1[300];
		
		softMax(outputLayer);
		
		for(int i = 0; i < 10; i++)
			deltaOfOut[i] = outputLayer[i].getOutput() - expectedValue[i];
		
		for(int j = 0; j < 100; j++){
			double sum = 0;
			for(int i = 0; i < 10; i++){
				sum += deltaOfOut[i] * hidden2[j].weights[i];
			}
			deltaOfH2[j] = sum * reluDerivFunctions(deltaOfH2[j].getOutput());
		}
		
		for(int j = 0; j < 300; j++){
			double sum = 0;
			for(int i = 0; i < 100; i++){
				sum += deltaOfOut[i] * hidden2[j].weights[i];
			}
			deltaOfH1[j] = sum * reluDerivFunctions(deltaOfH1[j].getOutput());
		}
		
		
		//update bias and weights
		for (int i = 0; i < 10; ++i) {
			outputLayer[i].bias -= learningRate * deltaOut[i];
			for (int j = 0; j < 100; ++j) {                      
				double grad = deltaOut[i] * hidden2[j].out;
				hidden2[j].weights[i] -= learningRate * grad;
			}
		}
		
		for (int k = 0; k < 300; ++k) {//bias problems
			for (int j = 0; j < 100; ++j) {
				double gradiant = deltaH2[j] * hidden1[k].out;
				hidden1[k].weights[j] -= learningRate * gradiant;
			}
			hidden1[k].bias -= learningRate * deltaH1[k];
		}
		
		for (int h = 0; h < 784; ++h) {
			for (int k = 0; k < 300; ++k) {
				double gradiant = deltaH1[k] * input[h].out;
				input[h].weights[k] -= learningRate * gradiant;
			}
		}
	}
};