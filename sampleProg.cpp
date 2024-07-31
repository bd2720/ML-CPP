#include <iostream>
#include "mdp.h"
#include "mdpValueIterator.h"
using namespace std;

int main(){
  cout << "Running \"Racing MDP\" Example" << endl;

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

  float discountRate = 1.0;
  MDPValueIterator vi(&problem, discountRate);
  cout << "MDP Value Iterator created; Discount = " << discountRate << endl;
  
  int iterations = 5;
  cout << "V*" << vi.getCurrentK() << "(s):";
  for(int s : vi.f_stateValue){
    cout << " " << s;
  }
  cout << endl;
  for(int i = 0; i < iterations; i++){
    // calculate next iteration
    vi.vIterate();
    // print values
    cout << "V*" << vi.getCurrentK() << "(s):";
    for(float s : vi.f_stateValue){
      cout << " " << s;
    }
    cout << endl;
  }

  return 0;
}