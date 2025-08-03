#ifndef FORWARDPROP_H_
#define FORWARDPROP_H_

#include <vector>
#include <iostream>
#include "Node.h"
#include "NNParams.h"

class ForwardProp {
public:
    void training(std::vector<Node>& inputLayer, 
                 std::vector<Node>& hidden1, 
                 std::vector<Node>& hidden2, 
                 std::vector<Node>& outputLayer,
                 std::vector<unsigned char>& image,
                 int numRows,
                 int numCols);

private:
    void forwardPropagation(std::vector<Node>& inputLayer, 
                           std::vector<Node>& hidden1, 
                           std::vector<Node>& hidden2, 
                           std::vector<Node>& outputLayer);
};

#endif
