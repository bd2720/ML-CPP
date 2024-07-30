#include <iostream>
#include "mdp.h"
#include "mdpValueIterator.h"
using namespace std;

int main(){
  cout << "Running sample..." << endl;

  int numStates = 3;
  int numActions = 2;
  MDP problem(numStates, numActions);
  cout << "Markov Decision Problem with " << numStates << " states and ";
  cout << numActions << " actions created." << endl;

  problem.addTransition(0, 0, 0, 1.0, 1);
  problem.addTransition(0, 1, 0, 0.5, 2);
  problem.addTransition(0, 1, 1, 0.5, 2);
  problem.addTransition(1, 0, 0, 0.5, 1);
  problem.addTransition(1, 0, 1, 0.5, 1);
  problem.addTransition(1, 1, 2, 1.0, -10);
  cout << "Transition function configured." << endl;

  MDPValueIterator vi(&problem); // create a Value Iterator to solve the MDP
  cout << "MDP Value Iterator created." << endl;

  return 0;
}