#include "Network.h"
#include <iostream>

int main(){
	std::cout<<"BASLIYORUZ"<<std::endl;
	
	Network neuralNetwork;
	std::cout << "--------------------------------" << std::endl;
	std::cout << "TRAINING" << std::endl;
	std::cout << "--------------------------------" << std::endl;
	neuralNetwork.trainTheNetwork();
	

	std::cout << "--------------------------------" << std::endl;
	std::cout << "TESTING" << std::endl;
	std::cout << "--------------------------------" << std::endl;

	neuralNetwork.testTheNetwork();

	std::cout << "--------------------------------" << std::endl;
	
	// Display all timing results
	neuralNetwork.displayAllTimes();
}