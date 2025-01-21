/* GOAL: select the most connected graph node
  (in case of a tie, choose first node)
*/

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include "fnn.hpp"
using namespace std;

#define GRAPH_NODES 10
// adjacency matrix representation of a graph
//    i connects to j and i < j ==> graph[i][j] == 1, else 0
int graph[GRAPH_NODES][GRAPH_NODES];

#define N_INPUTS ((GRAPH_NODES * (GRAPH_NODES - 1)) / 2)
#define N_OUTPUTS (GRAPH_NODES)
#define LAYERS {N_INPUTS, 100, N_OUTPUTS}
#define N_TRAINING 20000
#define N_EPOCHS 20

double inputs[N_INPUTS];
double expected[N_OUTPUTS];

// random binary (0 or 1)
int getRandomBinary(){
  return (rand() % 2);
}

// populate graph with random connections
void randomizeGraph(){
  for(int node1 = 0; node1 < GRAPH_NODES; node1++){
    for(int node2 = node1 + 1; node2 < GRAPH_NODES; node2++){
      graph[node1][node2] = getRandomBinary();
    }
  }
}

// loads graph into input array + populates expected; returns label.
// needs GRAPH_NODES size buffer for calculating most connected node
int loadGraph(){
  // load graph into inputs[]
  int inputIndex = 0;
  for(int node1 = 0; node1 < GRAPH_NODES; node1++){
    for(int node2 = node1 + 1; node2 < GRAPH_NODES; node2++){
      inputs[inputIndex++] = graph[node1][node2];
    }
  }

  // compute label (most connected node) in O(n^2)
  int numConnections[GRAPH_NODES] = {0};  
  for(int node1 = 0; node1 < GRAPH_NODES; node1++){
    for(int node2 = node1 + 1; node2 < GRAPH_NODES; node2++){
      // ignore non-connections
      if(graph[node1][node2] == 0) continue;
      // increment numConnections for both nodes
      numConnections[node1]++;
      numConnections[node2]++;
    }
  }
  // extract label (node with most connections)
  // if tied, lowest node will be selected
  int label = 0;
  for(int node = 0; node < GRAPH_NODES; node++){
    if(numConnections[node] > numConnections[label]){
      label = node;
    }
  }

  // initialize expected values from label
  for(int i = 0; i < N_OUTPUTS; i++){
    expected[i] = (i == label);
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
  FNN fnn(LAYERS, 0.1);
  fnn.initWeights(-0.3, 0.3);
  fnn.initBiases(0.0, 0.0);
  double *outputs = fnn.getOutputs();


  srand(time(0));
  auto trainingStart = chrono::high_resolution_clock::now();
  // training loop
  for(int epoch = 1; epoch <= N_EPOCHS; epoch++){
    int correct = 0;
    for(int example = 1; example <= N_TRAINING; example++){
      // create random graph
      randomizeGraph();
      // populate inputs with new graph, expected with label
      int label = loadGraph();

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
  }
  // stop timer
  auto trainingEnd = chrono::high_resolution_clock::now();
  auto trainingDuration = chrono::duration_cast<chrono::microseconds>(trainingEnd - trainingStart);
  
  double seconds = trainingDuration.count() / 1000000.0;
  cout << "Completed " << N_EPOCHS << " training epochs in ";
  cout << seconds << " seconds." << endl;
  cout << "Average seconds per epoch: " << seconds / N_EPOCHS << endl;
  cout << "Average seconds per example: " << seconds / (N_EPOCHS*N_TRAINING) << endl;

}

int main(){
  fnn_train();
}