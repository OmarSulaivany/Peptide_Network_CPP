#include "Neuron.h"
#include "Connection.h"
#include <vector>
#include <cmath>
#include <iostream>

// Overall net learning rate, we might need to tune this number to make our network perform better and faster.
double Neuron::learning_rate = 0.01;

// Momentum, multiplier of last deltaWeight, we might need to change this number to make our network perform better and faster.
double Neuron::momentum = 0.5;

Neuron::Neuron(unsigned numOutPuts, unsigned myIndex, unsigned numInPuts)
{

/* Loop through number of outputs, the next code will create connections based on the number of outputs, after that feed each weight
   with random number. */
 for (unsigned c = 0; c < numOutPuts; ++c)
 {
 	/* Create output connections in the neuron. */
 	m_outputWeights.push_back(Connection());

 	/* Set a random number to the Weight variable in the created connection. */
    m_outputWeights.back().weight = randomWeight(numInPuts);
  //  std::cout<<"N= "<<numInPuts<<"  Weight= "<<m_outputWeights.back().weight<<endl;
 }

/* Handle the index of the neuron locally. */
m_myIndex = myIndex;


}

double Neuron::Activation(double x)
{
    /* We will use Hyperbolic tangent function to transform our output value into a range between (-1,1). */
//	return tanh(x);
  //return 1 / (1 + exp(-x));
if (x > 0) return x;
	else return 0;
}

double Neuron::Activation_prime(double x)
{
    /* The derivative of Tanh(x). */
	//return 1- x * x;
  //return Activation(x) * (1 - Activation(x));

  if (x > 0) return 1;
	else return 0;
}



/* This function will return a random decimal number between 0,1. We need this function because when we first construct a neuron
 we need to assign to it random weight values. */
double Neuron::randomWeight(unsigned numInPuts)

{
  double rand_num;

  rand_num = -1 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (1 - (-1))));
  rand_num *=1/sqrt(numInPuts);
  return rand_num;

}



void Neuron::feedforward(const Layer &preLayer)
{
    // variable sum, to sum the output of each neuron of the previous layer including the bias neuron. (which are our inputs).
	double sum = 0.0;

    /* loop through each neuron in the previous layer including bias. */
	for(unsigned n = 0 ; n<preLayer.size(); ++n)
	{

		/* Sum all ( neurons * weights in the previous layer), and since our weight vector has it's own index we will pass m_myIndex
		wich is the current neuron index. */
		sum+= preLayer[n].getOutputVal() * preLayer[n].m_outputWeights[m_myIndex].weight;
	}

    /* Apply an activation function to the output to make it between a specific range, in our case -1,1. */
	m_outputVal = Neuron::Activation(sum);

	//m_outputVal = sum;


}

void Neuron::calcOutputGradients(double targetVals)
{
	/* Calculate the difference between the target and output value of the neuron. */
	double delta = targetVals - m_outputVal;

	/* To get the gradient we multiply the derivative of the activation function with our delta change. */
	m_gradient = delta * Neuron::Activation_prime(m_outputVal);
}

void Neuron::calcHiddenGradients(const Layer& nextLayer)
{
	/* Because the hidden layer doesn't have the Target value,and to calculate the error rate we will take the sum of the
	   derivative of weight of the next layer. */
	double dow = sumDOW(nextLayer);

	/* To get the gradient we multiply the derivative of the activation function with our delata change. */
	m_gradient = dow * Neuron::Activation_prime(m_outputVal);
}

double Neuron::sumDOW(const Layer& nextLayer) const
{
	// We will need this variable to store the summation of the derivative of weights of each neuron in the next layer.
	double sum = 0.0;

	/* loop through each neuron in the next layer except the bias neuron. */
	for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
	{
		/* The weights in the current neuron * the gradient decent of the neuron in the next layer. */
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::updateInputWeights(Layer& prevLayer)
{
	/* Note: The weights to be updated are in a container of connections in the neurons of the previous layer. */

	/* loop through each neuron in the previous layer including bias. */
	for (unsigned n = 0; n < prevLayer.size(); ++n)
	{
		/* Create a neuron to be same as the neuron in the previous layer. */
		Neuron& neuron = prevLayer[n];

		// Save all "current" weights of neuron of the previous layer.
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		 /* Factors that effects the updating of weights of each neuron are ("learning rate, gradient decent of the neuron and
		    the output value of the previous neuron.

		    learning rate * output of the previous neuron * gradient decent of the current neuron.
		    Plus momentum = a fraction of the previous delta weight * "Old Weights".

		    momentum helps to extend the movement in the same direction, when the oldweights are almost the same as the
		    current weights.*/
		double newDeltaWeight =

			/* Individual input, magnified by the gradient and train rate. */
			learning_rate

			/* Output value of the previous neuron. */
			* neuron.getOutputVal()

			// Gradient decent of the neuron
			* m_gradient

			/* momentum = a fraction of the previous delta weight. */
			+ momentum

			/* Old changing weight. */
			* oldDeltaWeight;

		/* Update the delta weights to store the current new weight. in the previous layer neurons. */
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;

		/* Update the current weights in each neuron of the previous layer, W_new=W_old + NewDeltaWeights. */
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
	}
}
