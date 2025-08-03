#pragma once
#include <vector>

class Node {
private:
    void randomValueGenerator(std::vector<double>& weights);

public:
    double input;
    double output;
    double bias;
    std::vector<double> weights;
    
    Node(int layerSize = 0);
    
    void setInput(const double input, const int neuronNumber);
    double getInput();
    double getOutput();
    void setBias(double bias);
    double getBias();
    void setOutput(double output);
};
