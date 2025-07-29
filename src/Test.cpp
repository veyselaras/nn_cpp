#include "ForwardProp.h"
#include "Test.h"

class Test{
public:

	void testTheNetwork(){
		std::string testFilename = "../train-images.idx3-ubyte";
		std::string testLabelFilename = "../train-labels.idx1-ubyte";
		
		std::ifstream testingFile(testFilename, std::ios::binary);
		std::ifstream testingLabel(testLabelFilename, std::ios::binary);
		
		if(testingFile && testingLabel)
			std::cout << "testing files is opened"<< std::endl;
		else{
			std::cout << "testing files is not opened"<< std::endl;
			return;
		}
		
		char magicNumber[4];
		char numberOfImages[4];
		char numberOfRows[4];
		char numberOfCols[4];
		
		testingFile.read(magicNumber, 4);
		testingFile.read(numberOfImages, 4);
		testingFile.read(numberOfRows, 4);
		testingFile.read(numberOfCols, 4);
		
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
	
		char magicNumber4Label[4];
		char numberOfLabels[4];
		testingLabel.read(magicNumber4Label, 4);
		testingLabel.read(numberOfLabels, 4);
		
		for(int i = 0; i < numImages; i++){
			std::vector<unsigned char> imageBytes(numRows * numCols);
			std::vector<unsigned char> labelByte(1);
			testingFile.read((char*)(imageBytes.data()), numRows * numCols);
			testingLabel.read((char*)(labelByte.data()), 1);
			
			forwardProp.training(inputLayer,
			                     hidden1Layer,
			                     hidden2Layer,
			                     outputLayer,
										imageBytes,
										numRows,
										numCols);
		}
	}
	
private:
};