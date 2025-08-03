#include "Network.h"
#include "Calculations.h"
#include "Test.h"
#include <chrono>

Network::Network() {
    // Initialize timing variables
    trainingTimeMs = 0.0;
    testTimeMs = 0.0;
    
    inputLayer.reserve(NNParams::INPUT);
    
    for (int i = 0; i < NNParams::INPUT; ++i)
        inputLayer.emplace_back(NNParams::INPUT);
    
    hidden1Layer.reserve(NNParams::H1);
    
    for (int i = 0; i < NNParams::H1; ++i)
        hidden1Layer.emplace_back(NNParams::H1);
    
    hidden2Layer.reserve(NNParams::H2);
    
    for (int i = 0; i < NNParams::H2; ++i)
        hidden2Layer.emplace_back(NNParams::H2);
    
    outputLayer.reserve(NNParams::OUTPUT);
    
    for (int i = 0; i < NNParams::OUTPUT; ++i)
        outputLayer.emplace_back(NNParams::OUTPUT);
}

void Network::trainTheNetwork() {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    std::string trainFilename = "./train-images.idx3-ubyte";
    std::string trainLabelFilename = "./train-labels.idx1-ubyte";
    
    std::ifstream trainingFile(trainFilename, std::ios::binary);
    std::ifstream trainingLabel(trainLabelFilename, std::ios::binary);
    
    if(trainingFile && trainingLabel)
        std::cout << "training file is opened"<< std::endl;
    else{
        std::cout << "training file is not opened"<< std::endl;
        return;
    }
    
    char numberOfImages[4];
    char numberOfRows[4];
    char numberOfCols[4];
    
    trainingFile.ignore(4);
    trainingFile.read(numberOfImages, 4);
    trainingFile.read(numberOfRows, 4);
    trainingFile.read(numberOfCols, 4);
    
    if(!trainingFile)
        std::cout << "HATA1" << std::endl;
    
    int numImages = (static_cast<unsigned char>(numberOfImages[0]) << 24) | 
                         (static_cast<unsigned char>(numberOfImages[1]) << 16) | 
                         (static_cast<unsigned char>(numberOfImages[2]) << 8) | 
                         static_cast<unsigned char>(numberOfImages[3]);
                         
    int numRows = (static_cast<unsigned char>(numberOfRows[0]) << 24) | 
                      (static_cast<unsigned char>(numberOfRows[1]) << 16) | 
                      (static_cast<unsigned char>(numberOfRows[2]) << 8) | 
                      static_cast<unsigned char>(numberOfRows[3]);
                      
    int numCols = (static_cast<unsigned char>(numberOfCols[0]) << 24) | 
                      (static_cast<unsigned char>(numberOfCols[1]) << 16) | 
                      (static_cast<unsigned char>(numberOfCols[2]) << 8) | 
                      static_cast<unsigned char>(numberOfCols[3]);


    char numberOfLabels[4];
    trainingLabel.ignore(4);
    trainingLabel.read(numberOfLabels, 4);
    
    int numLabels = (static_cast<unsigned char>(numberOfLabels[0]) << 24) | 
                         (static_cast<unsigned char>(numberOfLabels[1]) << 16) | 
                         (static_cast<unsigned char>(numberOfLabels[2]) << 8) | 
                          static_cast<unsigned char>(numberOfLabels[3]);
    
    if(numImages != numLabels)
        std::cout << "HATA SAYILAR UYUSMUYOR" << std::endl;
    
    for(int i = 0; i < numImages; i++){
        std::vector<unsigned char> imageBytes(numRows * numCols);
        std::vector<unsigned char> labelByte(1);
        trainingFile.read((char*)(imageBytes.data()), numRows * numCols);
        trainingLabel.read((char*)(labelByte.data()), 1);
        

        forwardProp.training(inputLayer,
                             hidden1Layer,
                             hidden2Layer,
                             outputLayer,
                             imageBytes,
                             numRows,
                             numCols);
        
        backProp.backPropagation(inputLayer,
                                hidden1Layer,
                                hidden2Layer,
                                outputLayer,
                                static_cast<int>(labelByte[0]));


        std::cout << "--------------------------------" << std::endl;
        std::cout << "Training completed in: " << i << " images" << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
    
    trainingFile.close();
    trainingLabel.close();
    
    // End timing and calculate duration
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Save training time
    trainingTimeMs = static_cast<double>(duration.count());
    
    std::cout << "Training completed in: " << duration.count() << " milliseconds" << std::endl;
    std::cout << "Training completed in: " << duration.count() / 1000.0 << " seconds" << std::endl;
}

void Network::testTheNetwork() {
    Test test;
    testTimeMs = test.testTheNetwork(inputLayer, hidden1Layer, hidden2Layer, outputLayer);
}

void Network::displayAllTimes() {
    std::cout << "==========================================" << std::endl;
    std::cout << "FINAL TIMING RESULTS" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Training Time: " << trainingTimeMs << " milliseconds (" << trainingTimeMs / 1000.0 << " seconds)" << std::endl;
    std::cout << "Testing Time:  " << testTimeMs << " milliseconds (" << testTimeMs / 1000.0 << " seconds)" << std::endl;
    std::cout << "Total Time:    " << (trainingTimeMs + testTimeMs) << " milliseconds (" << (trainingTimeMs + testTimeMs) / 1000.0 << " seconds)" << std::endl;
    std::cout << "==========================================" << std::endl;
}
