#include "Forward.h"
#include "Calculations.h"

class ForwardProp{
public:
	
	void training(std::vector<Node>& inputLayer, 
					  std::vector<Node>& hidden1, 
					  std::vector<Node>& hidden2, 
					  std::vector<Node>& outputLayer,
					  std::vector<unsigned char>& image,
					  int numRows,
					  int numCols)
	{
		size_t inputLayerSize = inputLayer.size();
		for(int i = 0: i < numCols * numRows; i++)
			inputLayer[i].setInput(image[i], inputLayerSize);
			
		forwardPropagation(inputLayer, hidden1Layer, hidden2Layer, outputLayer);
		
	}
	
private:

	Calculations calc;

	void forwardPropagation(std::vector<Node>& inputLayer, 
									std::vector<Node>& hidden1, 
									std::vector<Node>& hidden2, 
									std::vector<Node>& outputLayer)
	{
		calc.convolution(inputLayer, hidden1);
		calc.convolution(hidden1, hidden2);
		calc.convolution(hidden2, outputLayer);
	}
	
};