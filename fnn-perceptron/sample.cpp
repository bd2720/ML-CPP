#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "fnn.hpp"
using namespace std;

#define N_INPUTS 2
#define N_OUTPUTS 1
#define N_EXAMPLES 40000

double inputs[N_INPUTS];
double expected[N_OUTPUTS];

double getRandomDouble(double, double);
int getRandomBinary();

int main(){
  vector<int> layers = {N_INPUTS, 2, N_OUTPUTS};
  FNN fnn(layers);
  fnn.initWeights(0.5, 1.0);
  fnn.initBiases(0.0, 0.0);

  cout << "Layers: " << fnn.getNumLayers() << endl;
  for(int l = 0; l < fnn.getNumLayers(); l++){
    cout << "  Layer " << l << ": " << fnn.getNumNeurons(l) << " neurons" << endl;
  }

  srand(time(0));
  // train on N_EXAMPLES examples, computing SGD step for each one
  int correct = 0;
  for(int example = 1; example <= N_EXAMPLES; example++){
    // generate training example
    int a = getRandomBinary();
    int b = getRandomBinary();
    int y = (a ^ b);
    inputs[0] = (double)a;
    inputs[1] = (double)b;
    expected[0] = (double)y;

    // train fnn on example
    fnn.setInputs(inputs);
    fnn.computeActivations();
    fnn.backpropagate(expected);
    
    // evaluate correctness
    float ans = fnn.getOutputs()[0];
    correct += ((ans >= 0.5) ? 1 : 0) == y;
    // print accuracy periodically
    if(example % 100 == 1){
      cout << "#" << example << ": ";
      cout << ((double)correct) / ((double)example) << endl;
    }
  }
  return 0;
}

// random double within range
double getRandomDouble(double low, double high){
  return ((double) rand() / (double) RAND_MAX) * (high - low) + low;
}
// random binary (0 or 1)
int getRandomBinary(){
  return (rand() % 2);
}