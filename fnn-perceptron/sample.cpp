#include <iostream>
#include <vector>
#include "fnn.hpp"
using namespace std;

#define INPUTS 784

int main(){
  vector<int> layers = {INPUTS, 16, 16, 10};
  FNN fnn(layers);
  fnn.initWeights();
  fnn.initBiases();

  cout << "Layers: " << fnn.getNumLayers() << endl;
  cout << "Layer 0:" << endl;
  cout << "  Neurons: " << fnn.getNumNeurons(0) << endl;
  for(int l = 1; l < fnn.getNumLayers(); l++){
    cout << "Layer " << l << ":" << endl;
    cout << "  Neurons: " << fnn.getNumNeurons(l) << endl;
    cout << "w3,5 = " << fnn.getWeight(l, 3, 5) << endl;
    cout << "b3 = " << fnn.getBias(l, 3) << endl;
  }
  
  // create input array
  float inputs[INPUTS] = { 0 };
  fnn.setInputs(inputs);
  // set activations
  fnn.computeActivations();
  // extract outputs
  float *outputs = fnn.getOutputs();
  cout << "outputs: " << endl;
  for(int i = 0; i < fnn.getNumOutputs(); i++){
    cout << "  " << outputs[i] << endl;
  }
  return 0;
}
