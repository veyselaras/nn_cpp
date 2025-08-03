#include "Calculations.h"
#include <iostream>

double Calculations::reluFunc(double input){
    if(input <= 0){
        return 0;
    }
    else{
        return input;
    }
}

double Calculations::reluDerivFunc(double input){
    if(input <= 0){
        return 0;
    }
    else{
        return 1;
    }
}

void Calculations::softMax(std::vector<Node>& outputLayer){
    std::size_t vectorLength = outputLayer.size();
    double sumOfInputValues = 0;
    
    for(std::size_t i = 0; i < vectorLength; i++)
        sumOfInputValues += exp(outputLayer[i].getOutput());
    
    for(std::size_t i = 0; i < vectorLength; i++)
        outputLayer[i].setOutput(exp(outputLayer[i].getOutput())/sumOfInputValues);
}

void Calculations::convolution(std::vector<Node>& inputLayer, std::vector<Node>& outputLayer){
    std::size_t lengthOfInput = inputLayer.size();
    std::size_t lengthOfOutput = outputLayer.size();
    
    for(std::size_t i = 0; i < lengthOfOutput; i++){
        double sum = 0;
        for(std::size_t j = 0; j < lengthOfInput; j++){
            sum += inputLayer[j].weights[i]*inputLayer[j].getOutput();
        }
        outputLayer[i].setInput(sum + outputLayer[i].getBias(), lengthOfOutput);////////////////////////////////

    }
}