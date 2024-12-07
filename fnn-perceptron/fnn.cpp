#include <vector>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include "fnn.hpp"

using namespace std;

// random double within range
double randomDouble(double low, double high){
  return ((double) rand() / (double) RAND_MAX) * (high - low) + low;
}
// sigmoid activation function
static double sigmoid(double x){
  return 1.0 / (1.0 + exp(-x));
}
// derivative of sigmoid in terms of y (sigmoid)
static double dSigmoid(double y){
  return y * (1.0 - y);
}

FNN::FNN(const vector<int> &layers, double rate){
  // initialize number of layers
  this->numLayers = layers.size();
  this->learningRate = rate;
  this->numNeurons = layers;
  // resize vectors
  this->weight.resize(numLayers, nullptr);
  this->bias.resize(numLayers, nullptr);
  this->activation.resize(numLayers, nullptr);
  this->dActivation.resize(numLayers, nullptr);

  // allocate arrays for non-input layers
  for(int l = 1; l < numLayers; l++){
    // number of nodes in this layer
    int n = layers[l];
    // "2D" weights array for this layer (+ prev) 
    this->weight[l] = new double[n * layers[l-1]];
    // biases and activations for this layer's neurons
    this->bias[l] = new double[n];
    this->activation[l] = new double[n];
    this->dActivation[l] = new double[n];
  }
}

FNN::~FNN(){
  // free first layer activation (inputs)
  int l = 0;
  // free weight + bias + activation for each layer
  for(l = 1; l < numLayers; l++){
    delete[] this->weight[l];
    delete[] this->bias[l];
    delete[] this->activation[l];
    delete[] this->dActivation[l];
  }
}

void FNN::initWeights(double minWeight, double maxWeight){
  srand(time(0));
  // randomly initialize weights for non-input layers
  for(int l = 1; l < numLayers; l++){
    for(int n = 0; n < numNeurons[l]*numNeurons[l-1]; n++){
      weight[l][n] = randomDouble(minWeight, maxWeight);
    }
  }
}

void FNN::initBiases(double minBias, double maxBias){
  srand(time(0));
  // initialize weights on each non-input layer
  for(int l = 1; l < numLayers; l++){
    for(int n = 0; n < numNeurons[l]; n++){
      bias[l][n] = randomDouble(minBias, maxBias);
    }
  }
}

void FNN::feedforward(double *inputs){
  // set input activation
  activation[0] = inputs;
  // compute activation for all layers after input
  for(int l = 1; l < numLayers; l++){
    // calculate weighted sum for each neuron
    for(int j = 0; j < numNeurons[l]; j++){
      // add weight*activation for each neuron on previous layer
      double sum = getBias(l, j);
      for(int k = 0; k < numNeurons[l-1]; k++){
        sum += getWeight(l, j, k) * getActivation(l-1, k);
      }
      // set activation of jth neuron of lth layer
      activation[l][j] = sigmoid(sum);
    }
  }
}

void FNN::backpropagate(double *expected){
  // store dC0/dA calculated from cost in dActivation[outLayer]
  int outLayer = numLayers-1;
  for(int outNeuron = 0; outNeuron < getNumOutputs(); outNeuron++){
    dActivation[outLayer][outNeuron] = 2.0*(activation[outLayer][outNeuron] - expected[outNeuron]);
  }
  // propagate changes backwards, stop before input layer
  for(int l = outLayer; l > 0; l--){
    bool shouldComputeDelta = (l > 1);
    // 0-init all dActivation sums on previous layer
    if(shouldComputeDelta){ // skip dActivation for input layer
      for(int k = 0; k < numNeurons[l-1]; k++){
        dActivation[l-1][k] = 0;
      }
    }
    for(int j = 0; j < numNeurons[l]; j++){
      // save dC0/dZ
      double dCdZ = (dSigmoid(activation[l][j]) * dActivation[l][j]);
      // adjust bias
      bias[l][j] -= dCdZ * learningRate;
      // adjust of weights, accumulate previous activation adjustments
      for(int k = 0; k < numNeurons[l-1]; k++){
        // append to sum
        if(shouldComputeDelta) dActivation[l-1][k] += getWeight(l, j, k) * dCdZ;
        // set weight AFTER using in dAcivation sums
        decWeight(l, j, k, activation[l-1][k] * dCdZ * learningRate);
      }
    }
  }
}

double * FNN::getOutputs(){
  return activation[numLayers-1];
}