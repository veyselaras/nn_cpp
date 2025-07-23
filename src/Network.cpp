#include "Network.h"
#include "Node.h"
#include "ForwardProp.h"
#include "BackProp.h"
#include "NNParams.h"

class Network{
public:
	vector<Node> inputLayer;
	vector<Node> hidden1Layer;
	vector<Node> hidden2Layer;
	vector<Node> outputLayer;
	BackProp backProp;
	ForwardProp forwardProp;

	Network(){
		
		inputLayer.reserve(NNParams::INPUT);
		
		for (int i = 0; i < NNParams::INPUT; ++i)
			inputLayer.emplace_back(NNParams::INPUT);
		
		hidden1Layer.reserve(NNParams::H1);
		
		for (int i = 0; i < NNParams::H1; ++i)
			hidden1Layer.emplace_back(NNParams::H1);
		
		hidden2Layer.reserve(NNParams::H2);
		
		for (int i = 0; i < NNParams::H2; ++i)
			hidden2Layer.emplace_back(NNParams::H2);
		

		outputLayer.reserve(NNParams::OUTPUT);
		
		for (int i = 0; i < NNParams::OUTPUT; ++i)
			outputLayer.emplace_back(NNParams::OUTPUT);
	}

	void inputReading(){
		
	}
	
};