#include "Node.h"
#include "NNParams.h"
#include <iostream>
#include <random>
#include <ctime>

namespace {
    // dosya-genişliğinde tek rng
    std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
}

void Node::randomValueGenerator(std::vector<double>& w) {
    const std::size_t fan_in = w.size();         // her ağırlık = bir giriş
    const double sigma = std::sqrt(2.0 / fan_in);

    static thread_local std::mt19937 rng{std::random_device{}()};
    std::normal_distribution<double> dist(0.0, sigma);

    for (double& x : w)
        x = dist(rng);                           // He-normal
}

Node::Node(int layerSize) {
    std::size_t fanOut = 0;
    bias = 0.0; // Initialize bias to 0

    switch (layerSize)
    {
        case NNParams::INPUT:
            fanOut = NNParams::H1;
            break;
        case NNParams::H1:     
            fanOut = NNParams::H2;
            break;
        case NNParams::H2:     
            fanOut = NNParams::OUTPUT;
            break;
        case NNParams::OUTPUT: 
            fanOut = 0;
            break;
        default:
            std::cout<<"DEFATULT DEGER"<<std::endl;
            break;
    }

    weights.resize(fanOut);
    if (fanOut > 0) 
        randomValueGenerator(weights);
}

void Node::setInput(const double input, const int neuronNumber){
    this->input = input;
    if(neuronNumber == NNParams::H1 || neuronNumber == NNParams::H2){
        if(input>0)
            output = input;
        else
            output = 0;
    }
    else
        output = input;
}

double Node::getInput(){
    return input;
}

double Node::getOutput(){
    return output;
}

void Node::setBias(double bias){
    this->bias = bias;
}

double Node::getBias(){
    return bias;
}

void Node::setOutput(double output){
    this->output = output;
}