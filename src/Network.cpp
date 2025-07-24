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
































#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>


// Function to read IDX3-UBYTE files
std::vector<std::vector<unsigned char>> readIDX3UByteFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];
    char numRowsBytes[4];
    char numColsBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);
    file.read(numRowsBytes, 4);
    file.read(numColsBytes, 4);
    std::cout<<static_cast<int>(numImagesBytes[0])<<"  "<<static_cast<int>(numImagesBytes[1])<<"  "<<
        (int)static_cast<unsigned char>(numImagesBytes[2])<<"  "<<static_cast<int>(numImagesBytes[3])<<"  "<<std::endl;

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | static_cast<unsigned char>(numImagesBytes[3]);
    int numRows = (static_cast<unsigned char>(numRowsBytes[0]) << 24) | (static_cast<unsigned char>(numRowsBytes[1]) << 16) | (static_cast<unsigned char>(numRowsBytes[2]) << 8) | static_cast<unsigned char>(numRowsBytes[3]);
    int numCols = (static_cast<unsigned char>(numColsBytes[0]) << 24) | (static_cast<unsigned char>(numColsBytes[1]) << 16) | (static_cast<unsigned char>(numColsBytes[2]) << 8) | static_cast<unsigned char>(numColsBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(numRows * numCols);
        file.read((char*)(image.data()), numRows * numCols);

        images.push_back(image);
    }

    file.close();

    return images;
}

// Function to read IDX3-UBYTE files
std::vector<std::vector<unsigned char>> readLabelFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "Failed to open the IDX3-UBYTE file." << std::endl;
        return {};
    }

    // Read the IDX3-UBYTE file header
    char magicNumber[4];
    char numImagesBytes[4];

    file.read(magicNumber, 4);
    file.read(numImagesBytes, 4);

    // Convert the header information from big-endian to native endianness
    int numImages = (static_cast<unsigned char>(numImagesBytes[0]) << 24) | (static_cast<unsigned char>(numImagesBytes[1]) << 16) | (static_cast<unsigned char>(numImagesBytes[2]) << 8) | static_cast<unsigned char>(numImagesBytes[3]);

    // Initialize a vector to store the images
    std::vector<std::vector<unsigned char>> images;

    for (int i = 0; i < numImages; i++) {
        // Read each image as a vector of bytes
        std::vector<unsigned char> image(1);
        file.read((char*)(image.data()), 1);

        images.push_back(image);
    }

    file.close();

    return images;
}

int main() {

    std::string filename = "/home/cvlab/Downloads/train-images-idx3-ubyte/train-images.idx3-ubyte";
    std::string label_filename = "/home/cvlab/Downloads/train-labels-idx1-ubyte/train-labels.idx1-ubyte";

    std::vector<std::vector<unsigned char>> imagesFile = readIDX3UByteFile(filename);
    std::vector<std::vector<unsigned char>> labelsFile = readLabelFile(label_filename);
    std::vector<cv::Mat> imagesData;  // Store your images
    std::vector<int> labelsData;      // Corresponding labels

    for(int imgCnt=0; imgCnt<(int)imagesFile.size(); imgCnt++)
    {
        int rowCounter = 0;
        int colCounter = 0;

        cv::Mat tempImg = cv::Mat::zeros(cv::Size(28,28),CV_8UC1);
        for (int i = 0; i < (int)imagesFile[imgCnt].size(); i++) {

            tempImg.at<uchar>(cv::Point(colCounter++,rowCounter)) = (int)imagesFile[imgCnt][i];
            if ((i) % 28 == 0) {
                rowCounter++;
                colCounter= 0;
                if(i == 756)
                    break;
            }
        }
        std::cout<<(int)labelsFile[imgCnt][0]<<std::endl;

        imagesData.push_back(tempImg);
        labelsData.push_back((int)labelsFile[imgCnt][0]);

// to visualize each image ,n dataset  to check only
//        cv::imshow("TempImg",tempImg);
//        cv::waitKey(0);
    }


    return 0;
}