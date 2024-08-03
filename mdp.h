#ifndef MDP_H
#define MDP_H

#include <utility> // pair
#include <vector>  // vector
using namespace std;

// (probability, reward)
typedef pair<float, float> TransitionVal;

class MDP {
  private:
    int n_states;  // 0-indexed
    int n_actions; // 0-indexed
    vector<TransitionVal>  f_transition; // 3d vector indexed by idx(s, a, s')
    
    // index into 3d transition function
    int idx(int s, int a, int s2);

  public:

    // create a Markov Decision Problem with fixed states/actions
    MDP(int numStates, int numActions);
    // get n_states
    int getNumStates();
    // get n_actions;
    int getNumActions();
    // probability of s2 given (s, a)
    float getProbability(int s, int a, int s2);
    // reward for s2 given (s, a)
    float getReward(int s, int a, int s2);
    // register a new transition(s, a, s2) with probability p and reward r
    void addTransition(int s, int a, int s2, float p, float r);
};

#endif