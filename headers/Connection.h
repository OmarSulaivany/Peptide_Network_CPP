#ifndef CON_HH
#define CON_HH

struct Connection
{
    // initial weights.
	double weight;

	/* The change in the weights, we need this variable for momentum for example we can amplify momentum  if/
       weights update and delta weights update are in the same "the same or very similar". */
	double deltaWeight;
};
#endif
