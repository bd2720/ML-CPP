#include "mdp.h"
using namespace std;

/* PUBLIC */

MDP::MDP(int numStates, int numActions, float defCost){
  n_states = numStates;
  n_actions = numActions;
  defaultR = defCost;
  // allocate transition function
  int totalTransitions = n_states * n_states * n_actions;
  f_transition.resize(totalTransitions);
  // set default reward
  if(defaultR == 0.0) return;
  for(int i = 0; i < totalTransitions; i++){
    f_transition[i].second = defaultR;
  }
}

int MDP::getNumStates(){
  return this->n_states;
}

int MDP::getNumActions(){
  return this->n_actions;
}

float MDP::getProbability(int s, int a, int s2){
  return f_transition[idx(s, a, s2)].first;
}

float MDP::getReward(int s, int a, int s2){
  return f_transition[idx(s, a, s2)].second;
}

void MDP::addTransition(int s, int a, int s2, float prob, float reward){
  int transitionIndex = idx(s, a, s2);
  f_transition[transitionIndex].first = prob;
  f_transition[transitionIndex].second = reward;
}

void MDP::addTransition(int s, int a, int s2, float prob){
  int transitionIndex = idx(s, a, s2);
  f_transition[transitionIndex].first = prob;
}

/* PRIVATE */

int MDP::idx(int s, int a, int s2){
      return (s2 + n_states * (a + n_actions * (s)));
}