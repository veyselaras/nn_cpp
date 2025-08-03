#ifndef TEST_H_
#define TEST_H_

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "Node.h"
#include "ForwardProp.h"
#include "NNParams.h"

class Test {
public:
    double testTheNetwork(std::vector<Node>& inputLayer, 
                       std::vector<Node>& hidden1, 
                       std::vector<Node>& hidden2, 
                       std::vector<Node>& outputLayer);
};

#endif 