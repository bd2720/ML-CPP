#include <iostream>
#include <vector>
#include "fnn.hpp"
using namespace std;

#define N_INPUTS 1
#define N_OUTPUTS 1
#define N_EXAMPLES 100

float inputs[N_INPUTS] = {0.5};
float outputs[N_OUTPUTS] = {0.707};

int main(){
  vector<int> layers = {N_INPUTS, 16, 16, N_OUTPUTS};
  FNN fnn(layers);
  fnn.initWeights();
  fnn.initBiases();

  cout << "Layers: " << fnn.getNumLayers() << endl;
  for(int l = 0; l < fnn.getNumLayers(); l++){
    cout << "  Layer " << l << ": " << fnn.getNumNeurons(l) << " neurons" << endl;
  }

  // train on N_EXAMPLES examples, computing SGD step for each one
  for(int example = 0; example < N_EXAMPLES; example++){
    fnn.setInputs(inputs);
    fnn.computeActivations();
    fnn.backpropagate(outputs);
  }
  // show inputs
  cout << "inputs: " << endl;
  for(int i = 0; i < fnn.getNumInputs(); i++){
    cout << "  " << inputs[i] << endl;
  }
  // extract outputs
  float *outputs = fnn.getOutputs();
  cout << "outputs: " << endl;
  for(int i = 0; i < fnn.getNumOutputs(); i++){
    cout << "  " << outputs[i] << endl;
  }
  return 0;
}
