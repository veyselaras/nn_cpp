#include "Forward.h"
#include "Calculations.h"

class ForwardProp{
public:

	void forwardPropagation(std::vector<Node>& inputLayer, 
									std::vector<Node>& hidden1, 
									std::vector<Node>& hidden2, 
									std::vector<Node>& outputLayer)
	{
		convolution(inputLayer, hidden1);
		convolution(hidden1, hidden2);
		convolution(hidden2, outputLayer);
	}

};