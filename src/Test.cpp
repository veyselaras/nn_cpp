#include "ForwardProp.h"
#include "Test.h"

class Test{
public:

	void testTheNetwork(std::vector<Node>& inputLayer, 
							  std::vector<Node>& hidden1, 
							  std::vector<Node>& hidden2, 
							  std::vector<Node>& outputLayer){

		std::string testFilename = "../t10k-images.idx3-ubyte";
		std::string testLabelFilename = "../t10k-labels.idx1-ubyte";
		
		std::ifstream testingFile(testFilename, std::ios::binary);
		std::ifstream testingLabel(testLabelFilename, std::ios::binary);
		
		if(testingFile && testingLabel)
			std::cout << "testing files is opened"<< std::endl;
		else{
			std::cout << "testing files is not opened"<< std::endl;
			return;
		}
		
		char numberOfImages[4];
		char numberOfRows[4];
		char numberOfCols[4];
		
		testingFile.ignore(4);
		testingFile.read(numberOfImages, 4);
		testingFile.read(numberOfRows, 4);
		testingFile.read(numberOfCols, 4);
		
		int numImages = (static_cast<unsigned char>(numberOfImages[0]) << 24) | 
							 (static_cast<unsigned char>(numberOfImages[1]) << 16) | 
							 (static_cast<unsigned char>(numberOfImages[2]) << 8)  | 
							 static_cast<unsigned char>(numberOfImages[3]);
							 
		int numRows = (static_cast<unsigned char>(numberOfRows[0]) << 24) | 
						  (static_cast<unsigned char>(numberOfRows[1]) << 16) | 
						  (static_cast<unsigned char>(numberOfRows[2]) << 8) 	| 
						  static_cast<unsigned char>(numberOfRows[3]);
						  
		int numCols = (static_cast<unsigned char>(numberOfCols[0]) << 24) | 
						  (static_cast<unsigned char>(numberOfCols[1]) << 16) | 
						  (static_cast<unsigned char>(numberOfCols[2]) << 8) 	| 
						  static_cast<unsigned char>(numberOfCols[3]);
	
		char numberOfLabels[4];
		testingLabel.ignore(4);
		testingLabel.read(numberOfLabels, 4);
		
		int numLabels = (static_cast<unsigned char>(numberOfLabels[0]) << 24) | 
							 (static_cast<unsigned char>(numberOfLabels[1]) << 16) | 
							 (static_cast<unsigned char>(numberOfLabels[2]) << 8)  | 
							 static_cast<unsigned char>(numberOfLabels[3]);
		
		std::vector<unsigned char> imageBytes(numRows * numCols);
		
		if(numLabels != numImages){
			std::cout<<"numberOfLabels != numberOfImages"<<std::endl;
			return;
		}
		
		int correct = 0;
		
		for(int i = 0; i < numImages; i++){
			testingFile.read((char*)(imageBytes.data()), numRows * numCols);
			unsigned char label;                                        // (1)
			testingLabel.read(reinterpret_cast<char*>(&label), 1);
			
			forwardProp.training(inputLayer,
			                     hidden1,
			                     hidden2,
			                     outputLayer,
										imageBytes,
										numRows,
										numCols);
			
			int pred   = 0;                        // tahmin edilen sınıf
			double maxP = outputLayer[0].getOutput();

			for (int j = 1; j < NNParams::OUTPUT; ++j) {
				 double p = outputLayer[j].getOutput();
				 if (p > maxP) {
					  maxP = p;      // daha büyük olasılık
					  pred = j;      // → onun indisi sınıf tahmini
				 }
			}
			
			if (pred == label) ++correct;
		}
		testingFile.close();
		testingLabel.close();
		
		std::cout << "Accuracy: "
          << 100.0 * correct / numImages << "%\n";
	}
};