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
    std::vector<double*> weight;       // weight values, by [layer(1)][neuron1][neuron0]
    std::vector<double*> bias;         // bias, by [layer][neuron]
    std::vector<double*> activation;   // activation, by [layer][neuron]
    std::vector<double*> dActivation;  // partial sum storage for backprop, by [layer][neuron]
    double learningRate;               // coefficient of nudges to weights/biases

    // weight[l][n1][n0] -= x; for internal use by backpropagate()
    void decWeight(int layer, int neuron1, int neuron0, double x){
      weight[layer][W_IDX(neuron1, neuron0, numNeurons[layer-1])] -= x;
    }

  public:
    // initialize neural network (weights, biases, activation)
    // accepts a vector representing the number of neurons per layer
    FNN(std::vector<int> &layers, double rate = 0.1);
    // free dynamic arrays
    ~FNN();

    // getters
    int getNumLayers() { return numLayers; }
    int getNumNeurons(int layer) { return numNeurons[layer]; }
    double getWeight(int layer, int neuron1, int neuron0){ return weight[layer][W_IDX(neuron1, neuron0, numNeurons[layer-1])]; }
    double getBias(int layer, int neuron){ return bias[layer][neuron]; }
    double getActivation(int layer, int neuron){ return activation[layer][neuron]; }

    int getNumInputs(){ return numNeurons[0]; }
    int getNumOutputs(){ return numNeurons[numLayers-1]; }

    // initialize weight 2D arrays for each layer with [-maxWeight, maxWeight]
    void initWeights(double minWeight, double maxWeight);
    // initialize bias arrays for each layer
    void initBiases(double minBias, double maxBias);

    // copy double array to input layer (activation[0])
    void setInputs(double *inputs);
    // set activations + outputs assuming inputs are set
    void computeActivations();
    // backpropagate after a training example (+ adjust weights/biases)
    void backpropagate(double *expected);
    // expose output layer (activation[numLayers-1]) if activations were computed
    double *getOutputs();
};

#undef W_IDX
#endif