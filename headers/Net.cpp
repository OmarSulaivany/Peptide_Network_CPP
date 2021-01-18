#include "Net.h"
#include <iostream>
#include <cassert>
#include <vector>
#include<cmath>
using namespace std;

// Number of training samples to average over.
double Net::m_recentAverageSmoothingFactor = 100.0; 

/* pass the reference topology vector e.g {3,2,1}. */
Net::Net(const vector<unsigned> &topology)
{
    /* Total number of layers, we need it in order to create layers in the following line codes. */
	unsigned numLayers = topology.size();
 
     /* To create the corresponding layers per topology,/
	    we need this loop that starts from zero to the size of topology. */
	for(unsigned layerNum=0; layerNum<numLayers; ++layerNum)
	{   
	    /* Create a layer and add it to the m_layer 2d vector. */
		m_layers.push_back(Layer());

		/* numOutputs of layer[i] is the numInputs of layer[i+1] numOutputs of last layer is 0/
		numOutputs variable stores number of neurons in the next layer.
		
		This is important when we create neurons we will need to know
		how many output connections the neuron should have.
		this condition specifies the neurons in the last layer "output" layers
		has zero output connections.*/
		unsigned numOutPuts = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		
		
		
	    
	    /* We have made a new Layer, now fill it ith neurons, and
		add a bias neuron to the layer:
	    To add corresponding neurons per each layer,/
	    we need this loop that starts from zero to the size of layer plus 1 as we need one more neuron for the bias. */

		for(unsigned neuronNum=0 ; neuronNum <=topology[layerNum]; ++neuronNum)
		{
			cout<<" Layer "<<layerNum<<endl;
        /* Fill each layer with it's corresponding neurons, each neuron with it's number of output connections and it's index. */
		m_layers.back().push_back(Neuron(numOutPuts, neuronNum));
	     
		//cout<< "Connection weights " <<m_outputWeights.size()<<endl;
		//cout<<"===================================================\n";

		}

		/* Set the bias neuron output value to 1.0. which is the last neuron created in each layer. */
		m_layers.back().back().setOutputVal(1.0);
	}

}




void Net::feedForward(const vector <double> &inputVals)
{
    /* Assert function checks if the number of inputvals is the same as the number of input neurons
	in the first layer. if it's the same then everything is ok, if not it will raise an error message at run time.
	and the -1 at the end is we are excluding the bias neuron.*/

	// Check the num of inputVals euqal to neuronnum expect bias.
	assert(inputVals.size() == m_layers[0].size()-1);
 
    cout<<"Layer 0\n";
    /* loop through each neuron in the first layer and then assign inputvals into the neurons in the first layer. */
	for(unsigned i=0;i<inputVals.size();++i)
	{
		m_layers[0][i].setOutputVal(inputVals[i]);
		cout<<"Neuron "<<i<<" Output Value = "<<m_layers[0][i].getOutputVal()<<endl;

	}


    /* Forward propag. it means loop through each layer and then loop through each neuron inside that layer,
     after that tell each neuron to feedforward. we start from looping through the second layer because we've
     already set the first input layer with inputvals.*/
	for(unsigned layerNum = 1 ; layerNum<m_layers.size();++layerNum)
      {

      	cout<<"Layer number "<<layerNum<<endl;

      	/* Create new Layer object and point it to the previous layer. */
      	Layer &preLayer = m_layers[layerNum -1];

        /* loop through each neuron in the current layer, except the bias neuron. */
      	for(unsigned n = 0;n<m_layers[layerNum].size() -1;++n)
      	{
      		cout<<"Neuron "<<n<<" Output Value = ";
      		/* Apply feedforward function to each neuron. feedforward function is defined in the Neuron class. */
            m_layers[layerNum][n].feedforward(preLayer);

      	}
      	cout<<"===================================================\n";
      }
      cout<<"Net Layers size = "<<m_layers.size()<<endl;
	/* Note: when class Net is asked to feedforward, it's going to need to add up all of it's input values and then 
       apply a function to it to update it's output values. and in order to get the input values it needs to ask neurons
       in the previous layer what are the output values, because of that it is going to need to loop through all neurons 
  	   in the previous layer. */
}




void Net::backProbagation(const vector <double> &targetVals)
{

	/*Backpropagation function works as follow.
		1- Calculate the overall network error ( the difference between Output layer value with it's corresponding Target value )
	    by using some methods like MSE "mean sequare error", or RMS "Root mean Sequare error", MAE "Mean absloute error", binary 
	    crossentropy.... , we will use RMS.
		2- Calculate the gradient of the output layer.
		3- Calculate the gradient of the hidden "middle" layer/s.
		4- For every layer starting from output to the first hidden layer, update the connection weights*/


/* 1- Calculate overal net error (RMS of output neuron errors), RMS  Error = sqrt(1/n * sum(Target - output)^2). */

    /* Create a layer and the store the values of the last layer "output" layer in it. */
	Layer& outputLayer = m_layers.back(); 

	/* This member variable handles the error. */
	m_error = 0.0; 

    /* loop through each neurons in the output layer except the bias. */
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) 
	{
		/* delta = (Target - output) calculating the error for each single neuron in the output layer. */
		double delta = targetVals[n] - outputLayer[n].getOutputVal();

		/* Error = (delta)^2. power the error to get rid of negative result. also accumulate the error from the previous neuron
		   output value. */
		m_error += delta * delta;
	}
	/* Get average error squared. */
	m_error /= outputLayer.size() - 1; 

	/* RMS, sequare root the Error and by that we have the RMS fully applied. */
	m_error = sqrt(m_error); 

	/* Implement a recent average measurement, for visualizing "printing" the error in the last couple of iterations to see
	   how well the network perform. */
	m_recentAverageError =
		(m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
		/ (m_recentAverageSmoothingFactor + 1.0);

/* 2- Calcualting the Gradient decent of the last layer.Calculate output layer gradients. */

	/* loop through each neurons in the lastlayer except the bias. */
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) 
	{
		/* To calculate each neurons gradient it needs to have it's target value. */
		outputLayer[n].calcOutputGradients(targetVals[n]);
	}

/* 3- Calculating the Gradient decent of all neurons in the hidden Layers. */

    /* loop through all hidden layers . */
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
	{
		/* Create a Layer to store the current Hidden Layer. */
		Layer& hiddenLayer = m_layers[layerNum];  

        /* Create a Layer to store the Next Layer. to be the target outputs of the hiddent layer */
		Layer& nextLayer = m_layers[layerNum + 1];

		/* Loop through each neurons in the hidden layer except bias. */
		for (unsigned n = 0; n < hiddenLayer.size(); ++n) 
		{
			/* To calculate the gradient decent of each neuron in the hidden layer we will need to pass the Nextlayer. */
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

/* 4- for all layers starting from output to the first hidden layer, update the connection weights. */

	/* loop through all layers except the first and last layer as they dont have any weights. */
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
	{
		/* Create a layer to store the current layer. */
		Layer& layer = m_layers[layerNum]; 

		/* Create a layer to store previous layer. */
		Layer& prevLayer = m_layers[layerNum - 1]; 

		/* Loop through each neurons in the current layer except bias. */
		for (unsigned n = 0; n < layer.size() - 1; ++n) 
		{	
			/* To update the weights of each neurons we will need to pass prev-layer to it. */
			layer[n].updateInputWeights(prevLayer);
		}
	}
}





void Net::getResults(vector<double>& resultVals) const
{
	// Clear out the container.
	resultVals.clear();
    
    /* Loop through all the neuron in the output layer. */
	for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
	{   
		/* Move last layer neuron's output values into the resulVals. */
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}