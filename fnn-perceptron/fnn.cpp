#include <vector>
#include <ctime>
#include <cstdlib>

#include "fnn.hpp"

using namespace std;

// random float within range
static float getRandomFloat(float low, float high){
  return ((float) rand() / (float) RAND_MAX) * (high - low) + low;
}

FNN::FNN(vector<int> &layers){
  // initialize number of layers
  this->numLayers = layers.size();

  // configure input layer
  int n = layers[0];
  this->numNeurons.push_back(n);
  // input layer has no weight or bias
  this->weight.push_back(nullptr);
  this->bias.push_back(nullptr);
  // activation[0] has input neurons
  this->activation.push_back(new float[n]);

  // iterate through layers
  for(int l = 1; l < numLayers; l++){
    // number of nodes in this layer
    n = layers[l];
    this->numNeurons.push_back(n);
    // "2D" weights array for this layer (+ prev) 
    this->weight.push_back(new float[n * layers[l-1]]);
    // biases and activations for this layer's neurons
    this->bias.push_back(new float[n]);
    this->activation.push_back(new float[n]);
  }
}

FNN::~FNN(){
  // free first layer activation (inputs)
  int l = 0;
  delete[] this->activation[l];
  // free weight + bias + activation for each layer
  for(l = 1; l < numLayers; l++){
    delete[] this->weight[l];
    delete[] this->bias[l];
    delete[] this->activation[l];
  }
}

void FNN::initWeights(){
  srand(time(0));
  // seed RNG
  // randomly initialize weights for non-input layers
  for(int l = 1; l < numLayers; l++){
    for(int n = 0; n < numNeurons[l]*numNeurons[l-1]; n++){
      weight[l][n] = getRandomFloat(-0.1, 0.1);
    }
  }
}

void FNN::initBiases(){
  srand(time(0));
  // zero-initialize weights on each non-input layer
  for(int l = 1; l < numLayers; l++){
    for(int n = 0; n < numNeurons[l]; n++){
      bias[l][n] = 0;
    }
  }
}