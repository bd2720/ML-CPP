#include "mdp.h"
using namespace std;

MDP::MDP(int numStates, int numActions){
  n_states = numStates;
  n_actions = numActions;
  // allocate transition function
  int totalTransitions = n_states * n_states * n_actions;
  f_transition.resize(totalTransitions);
}

int MDP::idx(int s, int a, int s2){
      return (s2 + n_states * (a + n_actions * (s)));
}

void MDP::addTransition(int s, int a, int s2, float prob, float reward){
  int transitionIndex = idx(s, a, s2);
  f_transition[transitionIndex].first = prob;
  f_transition[transitionIndex].second = reward;
}

float MDP::getProbability(int s, int a, int s2){
  return f_transition[idx(s, a, s2)].first;
}

float MDP::getReward(int s, int a, int s2){
  return f_transition[idx(s, a, s2)].second;
}