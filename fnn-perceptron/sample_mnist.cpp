#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include "fnn.hpp"
using namespace std;

#define N_INPUTS 784
#define N_OUTPUTS 10
#define N_EXAMPLES 60000
#define N_EPOCHS 1
/* MNIST Data Source:
    https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
    Format: label, px1-1, px1-2, px1-3...
    label: int, 0-9 (digit)
    px: int, 0-255 (pixel brightness)
*/
const string mnist_training = "mnist_train.csv";

double inputs[N_INPUTS];
double expected[N_OUTPUTS];

// loads the next row into inputs and expected; returns label
int loadNextExample(ifstream &data){
  // create a string stream for the current row
  string row;
  getline(data, row);
  stringstream ss(row);

  // extract the label (0-9)
  int label;
  ss >> label;

  // initialize expected values from label
  for(int i = 0; i < N_OUTPUTS; i++){
    expected[i] = (i == label);
  }

  // extract and format inputs
  int brightness;
  for(int i = 0; i < N_INPUTS; i++){
    ss.ignore(); // skip comma
    ss >> brightness;
    // convert 0-255 pixel brightness into activation
    inputs[i] = brightness / 256.0;
  }
  return label;
}

// turn output array of digits into label (0-9)
int getOutputLabel(double *outputs){
  int maxIndex = 0;
  double max = outputs[0];
  for(int i = 1; i < N_OUTPUTS; i++){
    if(outputs[i] > max){
      max = outputs[i];
      maxIndex = i;
    }
  }
  return maxIndex;
}

void fnn_train(){
  // init model
  cout << "EXAMPLE 1 - MNIST TRAINING" << endl;
  FNN fnn({N_INPUTS, 16, 16, N_OUTPUTS});
  fnn.initWeights(-0.3, 0.3);
  fnn.initBiases(0.0, 0.0);
  double *outputs = fnn.getOutputs();

  // open mnist_train.csv
  ifstream trainingData(mnist_training);
  if(!trainingData.is_open()){
    cout << "Error: " << mnist_training << " could not be opened." << endl;
    return;
  }

  // training loop
  for(int epoch = 1; epoch <= N_EPOCHS; epoch++){
    int correct = 0;
    for(int example = 1; example <= N_EXAMPLES; example++){
      // load current MNIST example into inputs and expected arrays
      int label = loadNextExample(trainingData);

      // train on example
      fnn.feedforward(inputs);
      fnn.backpropagate(expected);

      
      // evaluate correctness (brightest neuron)
      int predictedLabel = getOutputLabel(outputs);
      correct += (predictedLabel == label);
      //cout << "Model Predicted: " << predictedLabel << "; Expected Value: " << label << endl;
    }
    // print accuracy
    cout << "e" << epoch << ": ";
    cout << ((double)correct) / ((double)N_EXAMPLES) * 100.0 << "%" << endl;
  }

  // close training data file
  trainingData.close();
}

int main(){
  fnn_train();
}