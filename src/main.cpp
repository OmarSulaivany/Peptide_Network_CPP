/* Neural Network Implementation in C++.
Author: Omar T. Mohammed
Date: 18-Jan-2021 */

#include <vector>
#include "../headers/Net.h"
#include <iostream>
using namespace std;


int main()

{
    /* Create a vector of topology, which specifies number of layers and neurons in the Neural network. */
	 vector <unsigned> topology;

	/* Input layer e.g topology.push_back(2) has 2 neurons */
	topology.push_back(2);

	/* Hidden layer, e.g topology.push_back(8) has 8 neurons */
	topology.push_back(8);

	/* Output layer, e.g topology.push_back(1) has one neuron*/
	topology.push_back(1);

 /* Initialize a Network with the above specifications */
  Net myNet(topology);

/* Creating a dynamic dataset to solve XOR problem. */

// First bit.
unsigned a=0;

//Second bit.
unsigned b=0;

// A vector to store inputs.
vector <double> inputVals;

// A vector to store Target outputs.
vector <double> targetVals;

// A vector to store Results after the training is finished.
vector <double> resultVals;


/***************************Training Phase**************************************/
/* do epochs-th to train the created Network. */
for(unsigned epochs=0;epochs<1000;++epochs)
{
	/* Store the first bit in the inputVals. */
	inputVals.push_back(a);

	/* Store the second bit in the inputVals. */
	inputVals.push_back(b);


/* These conditions assigns bits to a and b with it's corresponding Target to statisfy the XOR problem. */
	if(a==0 && b==0)
	{
		a=1;
		b=0;
		targetVals.push_back(0);
	}
	else if( a==1  && b==0)
	{
		a=0;
		b=1;
		targetVals.push_back(1);
	}
	else if( a==0  && b==1)
	{
		a=1;
		b=1;
		targetVals.push_back(1);
	}
	else if( a==1  && b==1)
	{
		a=0;
		b=0;
		targetVals.push_back(0);
	}

  /* Do feedForward. */
	myNet.feedForward(inputVals);

  /* Do Back backProbagation. */
	myNet.backProbagation(targetVals);

 /* Store the results in resultVals. */
	myNet.getResults(resultVals);

 /* Output the final resutls. */
	for(unsigned i=0;i<resultVals.size();++i)
		{cout<<"Prediction = "<<resultVals[i]<<endl;}

/* Clear all. */
	inputVals.clear();
	targetVals.clear();
	resultVals.clear();
}

//**********************Make Predictions************************************/

/* Clear the console. */
system("clear");

/* Enter the first bit. */
cout<<"Enter the first input:";
cin>>a;

/* Store the first bit into inputVals. */
inputVals.push_back(a);

/* Enter the Second bit. */
cout<<"Enter the Second input:";
cin>>b;

/* Store the first bit into inputVals. */
inputVals.push_back(b);

/* Do feedForward to get an output */
myNet.feedForward(inputVals);

/* Store the results in resultVals. */
myNet.getResults(resultVals);

/* Output the final resutl/s. */
for(unsigned i=0;i<resultVals.size();++i)
	{cout<<"Prediction = "<<resultVals[i]<<endl;}

    return 0;
}
