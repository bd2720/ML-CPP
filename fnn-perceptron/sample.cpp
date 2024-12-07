#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "fnn.hpp"
using namespace std;

#define N_INPUTS 3
#define N_OUTPUTS 1
#define N_EXAMPLES 100
#define N_EPOCHS 400

double inputs[N_INPUTS];
double expected[N_OUTPUTS];

double getRandomDouble(double, double);
int getRandomBinary();

int main(){
  // create FNN, specifying layers + (optional) learning rate
  const vector<int> layers = {N_INPUTS, 3, N_OUTPUTS};
  FNN fnn(layers, 0.1);

  cout << "Layers: " << fnn.getNumLayers() << endl;
  for(int l = 0; l < fnn.getNumLayers(); l++){
    cout << "  Layer " << l << ": " << fnn.getNumNeurons(l) << " neurons" << endl;
  }

  // initialize weights/biases randomly
  fnn.initWeights(0.5, 1.0);
  fnn.initBiases(0.0, 0.0);

  srand(time(0));
  // train on N_EXAMPLES examples, computing SGD step for each one
  int correct = 0;
  for(int epoch = 1; epoch <= N_EPOCHS; epoch++){
    correct = 0;
    for(int example = 1; example <= N_EXAMPLES; example++){
      // generate training example
      int a = getRandomBinary();
      int b = getRandomBinary();
      int c = getRandomBinary();
      int y = (a ^ b ^ c);
      inputs[0] = (double)a;
      inputs[1] = (double)b;
      inputs[2] = (double)c;
      expected[0] = (double)y;

      // train fnn on example
      fnn.feedforward(inputs);
      fnn.backpropagate(expected);

      // evaluate correctness
      float ans = fnn.getOutputs()[0];
      correct += ((ans >= 0.5) ? 1 : 0) == y;
    }
    // print/reset accuracy after each epoch
    cout << "e" << epoch << ": ";
    cout << ((double)correct) / ((double)N_EXAMPLES) * 100 << "%" << endl;
  }

  // print final weights
  cout << endl << "Final Weights:" << endl;
  for(int l = 1; l < fnn.getNumLayers(); l++){
    cout << "Layer " << l << ": " << endl;
    for(int j = 0; j < fnn.getNumNeurons(l); j++){
      for(int k = 0; k < fnn.getNumNeurons(l-1); k++){
        cout << "  w" <<  j << "," << k << " = " << fnn.getWeight(l, j, k) << endl;
      }
    }
  }
  // print final biases
  cout << endl << "Final Biases:" << endl;
  for(int l = 1; l < fnn.getNumLayers(); l++){
    cout << "Layer " << l << ": " << endl;
    for(int j = 0; j < fnn.getNumNeurons(l); j++){
      cout << "  b" <<  j << " = " << fnn.getBias(l, j) << endl;
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