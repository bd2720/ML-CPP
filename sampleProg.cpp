#include <iostream>
#include "mdp.h"
#include "mdpValueIterator.h"
using namespace std;

void sampleProblem1(){
  cout << "Running \"Racing\" MDP Example" << endl;

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

  MDPValueIterator vi(&problem); // discount rate 1.0 (default)
  cout << "MDP Value Iterator created; Discount = " << vi.getDiscount() << endl;
  
  int iterations = 5;
  int s; // state
  cout << "V*" << vi.getCurrentK() << "(s):";
  for(s = 0; s < problem.getNumStates(); s++){
    cout << " " << vi.getStateValue(s);
  }
  cout << endl;
  for(int i = 0; i < iterations; i++){
    // calculate next iteration
    vi.vIterate();
    // print values
    cout << "V*" << vi.getCurrentK() << "(s):";
    for(s = 0; s < problem.getNumStates(); s++){
      cout << " " << vi.getStateValue(s);
    }
    cout << endl;
  }

  // extract policy
  vi.extractPolicy();
  // display policy after last iteration
  cout << "pi" << vi.getCurrentK() << "(s):";
  for(int s = 0; s < problem.getNumStates(); s++){
    cout << " " << vi.getExtractedPolicy(s);
  }
  cout << endl;
}

void sampleProblem2(){
  cout << "Running \"Mars Rover\" MDP Example" << endl;

  int numStates = 2;
  int numActions = 2;
  MDP problem(numStates, numActions);
  cout << "Markov Decision Problem with " << numStates << " states and ";
  cout << numActions << " actions created." << endl;

  problem.addTransition(0, 0, 0, 0.8, 1);
  problem.addTransition(0, 0, 1, 0.2, 1);
  problem.addTransition(0, 1, 0, 0.6, 3);
  problem.addTransition(0, 1, 1, 0.4, 3);
  problem.addTransition(1, 0, 0, 0.7, -1);
  problem.addTransition(1, 0, 1, 0.3, -1);
  problem.addTransition(1, 1, 1, 1.0, 1);
  cout << "Transition function configured." << endl;

  float discountRate = 0.8;
  MDPValueIterator vi(&problem, discountRate);
  cout << "MDP Value Iterator created; Discount = " << vi.getDiscount() << endl;
  
  int iterations = 5;
  int s; // state
  cout << "V*" << vi.getCurrentK() << "(s):";
  for(s = 0; s < problem.getNumStates(); s++){
    cout << " " << vi.getStateValue(s);
  }
  cout << endl;
  for(int i = 0; i < iterations; i++){
    // calculate next iteration
    vi.vIterate();
    // print values
    cout << "V*" << vi.getCurrentK() << "(s):";
    for(s = 0; s < problem.getNumStates(); s++){
      cout << " " << vi.getStateValue(s);
    }
    cout << endl;
  }

  // extract policy
  vi.extractPolicy();
  // display policy after last iteration
  cout << "pi" << vi.getCurrentK() << "(s):";
  for(int s = 0; s < problem.getNumStates(); s++){
    cout << " " << vi.getExtractedPolicy(s);
  }
  cout << endl;
}

void sampleProblem3(){
  cout << "Running \"Hops\" MDP Example" << endl;

  MDP hops(3, 2);
  cout << "Markov Decision Problem with " << hops.getNumStates() << " states and ";
  cout << hops.getNumActions() << " actions created." << endl;

  hops.addTransition(0, 0, 0, 0.3, -1);
  hops.addTransition(0, 0, 1, 0.7, +2);
  hops.addTransition(0, 1, 0, 0.1, -1);
  hops.addTransition(0, 1, 1, 0.1, +1);

  hops.addTransition(1, 0, 0, 0.4, -2);
  hops.addTransition(1, 0, 2, 0.6, +3);
  hops.addTransition(1, 1, 1, 0.1, -1);
  hops.addTransition(1, 1, 2, 0.9, +1);

  hops.addTransition(2, 0, 0, 0.5, -3);
  hops.addTransition(2, 0, 2, 0.5, +10);
  hops.addTransition(2, 1, 0, 0.1, -10);
  hops.addTransition(2, 1, 2, 0.9, +1);
  cout << "Transition function configured." << endl;

  MDPValueIterator hopsVI(&hops, 0.9);
  cout << "MDP Value Iterator created; Discount = " << hopsVI.getDiscount() << endl;

  // INITIAL POLICY
  hopsVI.extractPolicy();
  cout << "pi" << hopsVI.getCurrentK() << "(s):";
  for(int s = 0; s < hops.getNumStates(); s++){
    cout << " " << hopsVI.getExtractedPolicy(s);
  }
  cout << endl;

  int iterations = 10;
  cout << "V*" << hopsVI.getCurrentK() << "(s):";
  for(int s = 0; s < hops.getNumStates(); s++){
    cout << " " << hopsVI.getStateValue(s);
  }
  cout << endl;
  for(int i = 0; i < iterations; i++){
    // calculate next iteration
    hopsVI.vIterate();
    // print values
    cout << "V*" << hopsVI.getCurrentK() << "(s):";
    for(int s = 0; s < hops.getNumStates(); s++){
      cout << " " << hopsVI.getStateValue(s);
    }
    cout << endl;
  }

  // FINAL POLICY
  hopsVI.extractPolicy();
  cout << "pi" << hopsVI.getCurrentK() << "(s):";
  for(int s = 0; s < hops.getNumStates(); s++){
    cout << " " << hopsVI.getExtractedPolicy(s);
  }
  cout << endl;
}

int main(){
  sampleProblem1();
  cout << endl;
  sampleProblem2();
  cout << endl;
  sampleProblem3();
  cout << endl;
  return 0;
}
