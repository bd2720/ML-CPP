#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include "fnn.hpp"
using namespace std;

#define N_INPUTS 784
#define N_OUTPUTS 10
#define N_TRAINING 60000
#define N_TESTING 10000
#define N_EPOCHS 5
/* MNIST Data Source:
    https://git-disl.github.io/GTDLBench/datasets/mnist_datasets/
    Format: label, px1-1, px1-2, px1-3...
    label: int, 0-9 (digit)
    px: int, 0-255 (pixel brightness)
*/
const string mnist_training = "mnist_train.csv";
const string mnist_testing = "mnist_test.csv";

double inputs[N_INPUTS];
double expected[N_OUTPUTS];

const string model_filename = "fnn_mnist.model";

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

  auto trainingStart = chrono::high_resolution_clock::now();
  // training loop
  for(int epoch = 1; epoch <= N_EPOCHS; epoch++){
    int correct = 0;
    for(int example = 1; example <= N_TRAINING; example++){
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
    // print accuracy and training time
    cout << "e" << epoch << ": ";
    cout << ((double)correct) / ((double)N_TRAINING) * 100.0 << "%" << endl;

    // reset training data
    trainingData.seekg(0);
  }
  // stop timer
  auto trainingEnd = chrono::high_resolution_clock::now();
  auto trainingDuration = chrono::duration_cast<chrono::microseconds>(trainingEnd - trainingStart);
  
  double seconds = trainingDuration.count() / 1000000.0;
  cout << "Completed " << N_EPOCHS << " training epochs in ";
  cout << seconds << " seconds." << endl;
  cout << "Average seconds per epoch: " << seconds / N_EPOCHS << endl;
  cout << "Average seconds per example: " << seconds / (N_EPOCHS*N_TRAINING) << endl;

  // close training data file
  trainingData.close();

  // export model
  fnn.exportParameters(model_filename);
}

// load model from file and test on testing set
void fnn_test(){
  cout << "EXAMPLE 2 - MNIST TESTING" << endl;
  FNN fnn({N_INPUTS, 16, 16, N_OUTPUTS});
  fnn.importParameters(model_filename); // import from file
  double *outputs = fnn.getOutputs();

  // open testing set
  ifstream testingData(mnist_testing);
  if(!testingData.is_open()){
    cout << "Error: " << mnist_testing << " could not be opened." << endl;
    return;
  }

  // testing loop
  int correct = 0;
  for(int example = 0; example < N_TESTING; example++){
    // load example
    int label = loadNextExample(testingData);

    // compute activations
    fnn.feedforward(inputs);

    // evaluate correctness
    int predictedLabel = getOutputLabel(outputs);
    correct += (predictedLabel == label);
  }

  // print testing accuracy
  cout << "Testing Accuracy: ";
  cout << ((double)correct) / ((double)N_TRAINING) * 100.0 << "%";
  cout << " (" << correct << "/" << N_TRAINING << ")" << endl;
}

int main(){
  fnn_train();
  fnn_test();
}