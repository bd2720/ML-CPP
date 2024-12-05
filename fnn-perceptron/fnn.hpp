#ifndef FNN_H
#define FNN_H
#include <vector>

/*  Feedforward Neural Network, or "Multilayer Perceptron"
    Inspired by:
      3Blue1Brown's course on neural networks
      Nicolai Nielsen's "Coding a Neural Network from Scratch in C"
*/

// index into weights based on neuron1, neuron0
#define W_IDX(n1, n0, l0) ((n1) * (l0)) + (n0)

class FNN {
  private:
    int numLayers;                    // number of layers, incl. output
    std::vector<int> numNeurons;      // number of neurons, by layer
    std::vector<float*> weight;       // weight values, by [layer(1)][neuron1][neuron0]
    std::vector<float*> bias;         // bias, by [layer][neuron]
    std::vector<float*> activation;   // activation, by [layer][neuron]
    std::vector<float*> dActivation;  // partial sum storage for backprop, by [layer][neuron]
    float learningRate;               // coefficient of nudges to weights/biases

    // weight[l][n1][n0] -= x; for internal use by backpropagate()
    void decWeight(int layer, int neuron1, int neuron0, float x){
      weight[layer][W_IDX(neuron1, neuron0, numNeurons[layer-1])] -= x;
    }

  public:
    // initialize neural network (weights, biases, activation)
    // accepts a vector representing the number of neurons per layer
    FNN(std::vector<int> &layers, float rate = 0.1);
    // free dynamic arrays
    ~FNN();

    // getters
    int getNumLayers() { return numLayers; }
    int getNumNeurons(int layer) { return numNeurons[layer]; }
    float getWeight(int layer, int neuron1, int neuron0){ return weight[layer][W_IDX(neuron1, neuron0, numNeurons[layer-1])]; }
    float getBias(int layer, int neuron){ return bias[layer][neuron]; }
    float getActivation(int layer, int neuron){ return activation[layer][neuron]; }

    int getNumInputs(){ return numNeurons[0]; }
    int getNumOutputs(){ return numNeurons[numLayers-1]; }

    // initialize weight 2D arrays for each layer
    void initWeights();
    // initialize bias arrays for each layer
    void initBiases();

    // copy float array to input layer (activation[0])
    void setInputs(float *inputs);
    // set activations + outputs assuming inputs are set
    void computeActivations();
    // backpropagate after a training example (+ adjust weights/biases)
    void backpropagate(float *expected);
    // expose output layer (activation[numLayers-1]) if activations were computed
    float *getOutputs();
};

#undef W_IDX
#endif