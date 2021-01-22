/* Neural Network Implementation in C++.
Author: Omar T. Mohammed
Date: 18-Jan-2021 */

//g++ -include ../headers/Net.cpp -include ../headers/Neuron.cpp main.cpp -o main `pkg-config --cflags --libs opencv`

//! [includes]
#include "../headers/Net.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <algorithm>
#include <vector>
#include <iostream>
using namespace cv;
using namespace std;


int main(void)

{
    /* Create a vector of topology, which specifies number of layers and neurons in the Neural network. */
	 vector <unsigned> topology;

	/* Input layer e.g topology.push_back(2) has 2 neurons */
	topology.push_back(784);

	/* Hidden layer, e.g topology.push_back(8) has 8 neurons */
	topology.push_back(250);

	/* Hidden layer, e.g topology.push_back(8) has 8 neurons */
	topology.push_back(127);

	/* Hidden layer, e.g topology.push_back(8) has 8 neurons */
	topology.push_back(64);

	/* Hidden layer, e.g topology.push_back(8) has 8 neurons */
	topology.push_back(32);

	/* Output layer, e.g topology.push_back(1) has one neuron*/
	topology.push_back(1);

 /* Initialize a Network with the above specifications */
  Net myNet(topology);


// A vector to store inputs.
vector <double> inputVals;

// A vector to store Target outputs.
vector <double> targetVals;

// A vector to store Results after the training is finished.
vector <double> resultVals;

	srand(time(0));
/***************************Training Phase**************************************/
/* do epochs-th to train the created Network. */
for(unsigned epochs=0;epochs<10000;++epochs)
{

	// Read all .jpg files from the specified folder

		double target=rand()%2;
		char b = target +'0';
		string image_path = "../images/";
		image_path.append(sizeof(char), b);
		image_path+="/*.jpg";
		vector<string> filenames;
		glob(image_path, filenames);
		//Random shuffle
		random_shuffle(filenames.begin(), filenames.end());
		Mat img = imread(filenames[0], IMREAD_COLOR);
		//! [imread]

		//! [empty]
		if(img.empty())
		{
				cout << "Could not read the image: " << image_path << std::endl;
				return 1;
	 }
	 resize(img, img, Size(28,28), 0, 0, INTER_CUBIC); // resize to 28x28 resolution
	 cvtColor(img, img, COLOR_BGR2GRAY);


	img.convertTo(img, CV_32FC1); // or CV_32F works (too)
	//vector<double> array;
	if (img.isContinuous()) {
	inputVals.assign((float*)img.datastart, (float*)img.dataend);
	} else {
	for (int i = 0; i < img.rows; ++i) {
		inputVals.insert(inputVals.end(), img.ptr<float>(i), img.ptr<float>(i)+img.cols);

	}
	}


  /* Do feedForward. */
	myNet.feedForward(inputVals);

  /* Do Back backProbagation. */
	targetVals.push_back(target);
	myNet.backProbagation(targetVals);

 /* Store the results in resultVals. */
	myNet.getResults(resultVals);

 /* Output the final resutls. */
	//for(unsigned i=0;i<resultVals.size();++i)
		{cout<<"Iteration: "<<epochs<<endl<<"Real = "<<target<<"			Prediction = "<<resultVals[i]<<endl;
		 cout<<"===============================================================\n\n";}

/* Clear all. */
	inputVals.clear();
	targetVals.clear();
	resultVals.clear();
}

//**********************Make Predictions************************************/

/* Clear the console. */
//system("clear");

string image_path = "../images/tests/test2.jpg";
Mat img = imread(image_path, IMREAD_COLOR);
//! [imread]

//! [empty]
if(img.empty())
{
		std::cout << "Could not read the image: " << image_path << std::endl;
		return 1;
}
resize(img, img, Size(28,28), 0, 0, INTER_CUBIC); // resize to 28x28 resolution
cvtColor(img, img, COLOR_BGR2GRAY);


img.convertTo(img, CV_32FC1); // or CV_32F works (too)

if (img.isContinuous()) {
inputVals.assign((float*)img.datastart, (float*)img.dataend);
} else {
for (int i = 0; i < img.rows; ++i) {
inputVals.insert(inputVals.end(), img.ptr<float>(i), img.ptr<float>(i)+img.cols);

}
}

/* Do feedForward to get an output */
myNet.feedForward(inputVals);

/* Store the results in resultVals. */
myNet.getResults(resultVals);

/* Output the final resutl/s. */
for(unsigned i=0;i<resultVals.size();++i)
	{cout<<"Prediction = "<<resultVals[i]<<endl;}

    return 0;
}
