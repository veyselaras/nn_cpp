#include "ForwardProp.h"
#include "Calculations.h"

void ForwardProp::training(std::vector<Node>& inputLayer, 
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
        inputLayer[i].setInput(static_cast<double>(image[i])/255.0, inputLayerSize);
        
    forwardPropagation(inputLayer, hidden1, hidden2, outputLayer);
}

void ForwardProp::forwardPropagation(std::vector<Node>& inputLayer, 
                                    std::vector<Node>& hidden1, 
                                    std::vector<Node>& hidden2, 
                                    std::vector<Node>& outputLayer)
{
    Calculations calc;
    
    // Hidden layer 1 with ReLU
    calc.convolution(inputLayer, hidden1);
    
    // Hidden layer 2 with ReLU
    calc.convolution(hidden1, hidden2);
    calc.convolution(hidden2, outputLayer);
    
    // Apply softmax to output layer
    calc.softMax(outputLayer);
}