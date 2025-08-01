#include "Forward.h"
#include "Calculations.h"
#include "NNParams.h"

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
		if (image.size() != NNParams::INPUT) 
		{
			std::cout<<"image.size() != NNParams::INPUT"<<std::endl;
		}
		size_t inputLayerSize = inputLayer.size();
		for(int i = 0; i < image.size(); i++)
			inputLayer[i].setInput(image[i]/255.0, inputLayerSize);
			
		forwardPropagation(inputLayer, hidden1, hidden2, outputLayer);
		
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
		
		calc.softMax(outputLayer);
	}
	
};