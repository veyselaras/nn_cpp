#include "BackProp.h"
#include "Calculations.h"
#include "NNParams.h"

class BackProp{
public:

		void backPropagation(std::vector<Node>& inputLayer, 
									std::vector<Node>& hidden1, 
									std::vector<Node>& hidden2, 
									std::vector<Node>& outputLayer, 
									std::vector<unsigned char> labelByte){
		
		double deltaOfOut[NNParams::OUTPUT];
		double deltaOfH2[NNParams::H2];
		double deltaOfH1[NNParams::H1];
		
		for(int i = 0; i < NNParams::OUTPUT; i++)
			deltaOfOut[i] = outputLayer[i].getOutput() - static_cast<int>(labelByte[i]);
		
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