#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include "fnn.hpp"

using namespace std;

// random float within range
static float getRandomFloat(float low, float high){
  return ((float) rand() / (float) RAND_MAX) * (high - low) + low;
}
// sigmoid activation function
static float sigmoid(float x){
  return 1.0 / (1.0 + exp(-x));
}
// derivative of sigmoid in terms of y (sigmoid)
static float dSigmoid(float y){
  return y * (1.0 - y);
}

FNN::FNN(vector<int> &layers, float rate){
  // initialize number of layers
  this->numLayers = layers.size();
  this->learningRate = rate;

  // configure input layer
  int n = layers[0];
  this->numNeurons.push_back(n);
  // input layer has no weight or bias
  this->weight.push_back(nullptr);
  this->bias.push_back(nullptr);
  // activation[0] has input neurons
  this->activation.push_back(new float[n]);
  // can't propagate back to input neurons (fixed)
  this->dActivation.push_back(nullptr);

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
    this->dActivation.push_back(new float[n]);
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
    delete[] this->dActivation[l];
  }
}

void FNN::initWeights(){
  srand(time(0));
  // seed RNG
  // randomly initialize weights for non-input layers
  for(int l = 1; l < numLayers; l++){
    for(int n = 0; n < numNeurons[l]*numNeurons[l-1]; n++){
      weight[l][n] = getRandomFloat(-0.28, 0.28);
    }
  }
}

void FNN::initBiases(){
  // zero-initialize weights on each non-input layer
  for(int l = 1; l < numLayers; l++){
    for(int n = 0; n < numNeurons[l]; n++){
      bias[l][n] = 0;
    }
  }
}

void FNN::setInputs(float *inputs){
  // assumes inputs is not null and contains numNeurons[0] vals
  for(int n = 0; n < numNeurons[0]; n++){
    activation[0][n] = inputs[n];
  }
}

void FNN::computeActivations(){
  // compute activation for all layers after input
  for(int l = 1; l < numLayers; l++){
    // calculate weighted sum for each neuron
    for(int j = 0; j < numNeurons[l]; j++){
      // add weight*activation for each neuron on previous layer
      float sum = getBias(l, j);
      for(int k = 0; k < numNeurons[l-1]; k++){
        sum += getWeight(l, j, k) * getActivation(l-1, k);
      }
      // set activation of jth neuron of lth layer
      activation[l][j] = sigmoid(sum);
    }
  }
}

void FNN::backpropagate(float *expected){
  // store dC0/dA calculated from cost in dActivation[outLayer]
  int outLayer = numLayers-1;
  for(int outNeuron = 0; outNeuron < getNumOutputs(); outNeuron++){
    dActivation[outLayer][outNeuron] = 2.0*(activation[outLayer][outNeuron] - expected[outNeuron]);
  }
  // propagate changes backwards, stop before input layer
  for(int l = numLayers-1; l > 0; l--){
    // 0-init all dActivation sums on previous layer
    if(l > 1){ // skip dActivation if
      for(int k = 0; k < numNeurons[l-1]; k++){
        dActivation[l-1][k] = 0;
      }
    }
    for(int j = 0; j < numNeurons[l]; j++){
      // save dC0/dZ
      float dCdZ = (dSigmoid(activation[l][j]) * dActivation[l][j]);
      // adjust bias
      bias[l][j] -= dCdZ * learningRate;
      // adjust of weights, accumulate previous activation adjustments
      for(int k = 0; k < numNeurons[l-1]; k++){
        // append to sum
        if(l > 1) dActivation[l-1][k] += getWeight(l, j, k) * dCdZ;
        // set weight AFTER using in dAcivation sums
        decWeight(l, j, k, activation[l-1][k] * dCdZ * learningRate);
      }
    }
  }
}

float * FNN::getOutputs(){
  return activation[numLayers-1];
}