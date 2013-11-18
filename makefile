all: stitching shotanalysis

stitching: opticalflowstitching.cpp
	g++ opticalflowstitching.cpp -o bin/optical `pkg-config --libs --cflags opencv` -std=c++0x -pedantic -Wall

shotanalysis: shotanalysis.cpp
	g++ shotanalysis.cpp -o bin/shots `pkg-config --libs --cflags opencv` -std=c++0x -pedantic -Wall

