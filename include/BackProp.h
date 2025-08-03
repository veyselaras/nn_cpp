#ifndef BACKPROP_H_
#define BACKPROP_H_

#include <vector>
#include "Node.h"
#include "NNParams.h"

class BackProp {
public:
    void backPropagation(std::vector<Node>& inputLayer, 
                        std::vector<Node>& hidden1, 
                        std::vector<Node>& hidden2, 
                        std::vector<Node>& outputLayer, 
                        int labelByte);
};

#endif
