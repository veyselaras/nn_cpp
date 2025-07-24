#include "Network.h"
#include "Node.h"
#include "ForwardProp.h"
#include "BackProp.h"
#include "NNParams.h"
#include <fstream>


class Network{
public:
	vector<Node> inputLayer;
	vector<Node> hidden1Layer;
	vector<Node> hidden2Layer;
	vector<Node> outputLayer;
	BackProp backProp;
	ForwardProp forwardProp;

	Network(){
		
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
	
	void trainTheNetwork(){
		std::string trainFilename = "../t10k-images.idx3-ubyte";
		std::string trainLabelFilename = "../t10k-labels.idx1-ubyte";
		std::string testFilename = "../train-images.idx3-ubyte";
		std::string testLabelFilename = "../train-labels.idx1-ubyte";
		
		std::ifstream trainingFile(trainFilename, std::ifstream::binary);
		std::ifstream trainingLabel(trainLabelFilename, std::ifstream::binary);
		
		if(trainingFile & trainingLabel)
			std::cout << "training file is opened"<< std::endl;
		else{
			std::cout << "training file is opened"<< std::endl;
			return;
		}
		
		char magicNumber[4];
		char numberOfImages[4];
		char numberOfRows[4];
		char numberOfCols[4];
		
		file.read(magicNumber, 4);
		file.read(numberOfImages, 4);
		file.read(numberOfRows, 4);
		file.read(numberOfCols, 4);
		
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
	
		char magicNumber4Label[4]
		
		for(int i = 0; i < numImages; i++){
			std::vector<unsigned char> imageBytes(numRows * numCols);
			std::vector<unsigned char> labelByte;
			file.read((char*)(imageBytes.data()), numRows * numCols);
			file.read((char*)(labelByte.data()), 1);
			
			forwardProp.training(inputLayer,
			                     hidden1Layer,
			                     hidden2Layer,
			                     outputLayer,
										imageBytes,
										numRows,
										numCols)
			
			backProp.backPropagation(	inputLayer,
												hidden1Layer,
												hidden2Layer,
												outputLayer,
												labelByte
											);
			
			
		}
		
	}
};
