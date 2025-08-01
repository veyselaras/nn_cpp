#pragma once
#include <vector>

class Node {
public:
    unsigned char input;
    double output;
    double bias = 0;
    std::vector<double> weights;
    
    Node(int layerSize = 0);
    
    void setInput(const unsigned char input, const size_t neuronNumber);
    unsigned char getInput();
    double getOutput();
    void setBias(double output);
    double getBias();
    void setOutput(double output);
};
