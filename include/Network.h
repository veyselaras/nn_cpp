#ifndef NETWORK_H_
#define NETWORK_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include "Node.h"
#include "ForwardProp.h"
#include "BackProp.h"
#include "NNParams.h"

class Network {
public:
    std::vector<Node> inputLayer;
    std::vector<Node> hidden1Layer;
    std::vector<Node> hidden2Layer;
    std::vector<Node> outputLayer;
    BackProp backProp;
    ForwardProp forwardProp;
    
    // Timing storage
    double trainingTimeMs;
    double testTimeMs;

    Network();
    void trainTheNetwork();
    void testTheNetwork();
    void displayAllTimes();
};

#endif
