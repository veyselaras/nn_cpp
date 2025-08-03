all: compile run
nesneler:= ./lib/Test.o ./lib/Network.o ./lib/BackProp.o ./lib/ForwardProp.o ./lib/NNParams.o ./lib/Calculations.o ./lib/Node.o

compile:
	g++ -I ./include/ -o ./lib/NNParams.o -c ./src/NNParams.cpp
	g++ -I ./include/ -o ./lib/Node.o -c ./src/Node.cpp
	g++ -I ./include/ -o ./lib/Calculations.o -c ./src/Calculations.cpp
	g++ -I ./include/ -o ./lib/ForwardProp.o -c ./src/ForwardProp.cpp
	g++ -I ./include/ -o ./lib/BackProp.o -c ./src/BackProp.cpp
	g++ -I ./include/ -o ./lib/Network.o -c ./src/Network.cpp
	g++ -I ./include/ -o ./lib/Test.o -c ./src/Test.cpp
	g++ -I ./include/ -o ./bin/main $(nesneler) ./src/main.cpp
run:
	./bin/main