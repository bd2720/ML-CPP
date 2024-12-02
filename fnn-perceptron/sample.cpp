#include <iostream>
#include <vector>
#include "fnn.hpp"
using namespace std;

int main(){
  vector<int> layers = {784, 16, 16, 10};
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
  return 0;
}
