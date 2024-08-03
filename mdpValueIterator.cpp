#include "mdp.h"
#include "mdpValueIterator.h"
using namespace std;

/* PUBLIC */

MDPValueIterator::MDPValueIterator(MDP * mdpRef, float discountRate){
  this->mdp = mdpRef;
  this->mdpStates = mdpRef->getNumStates();
  this->mdpActions = mdpRef->getNumActions();
  this->discount = discountRate;
  k = 0; // 0th iteration
  f_stateValue.resize(mdpStates);
  f_prevStateValue.resize(mdpStates);
  f_qStateValue.resize(mdpStates * mdpActions);
}

float MDPValueIterator::getDiscount(){
  return this->discount;
}

int MDPValueIterator::getCurrentK(){
  return this->k;
}

float MDPValueIterator::getQStateValue(int s, int a){
  return this->f_qStateValue[q_idx(s, a)];
}

float MDPValueIterator::getStateValue(int s){
  return this->f_stateValue[s];
}

void MDPValueIterator::setQStateValue(int s, int a, float val){
  this->f_qStateValue[q_idx(s, a)] = val;
}

void MDPValueIterator::setStateValue(int s, float val){
  this->f_stateValue[s] = val;
}

/* PRIVATE */

int MDPValueIterator::q_idx(int s, int a){
  return a + mdpActions * s;
}

float MDPValueIterator::calcQStateValue(int s, int a){
  float sum = 0.0;
  // sum together value over all destination states
  for(int s2 = 0; s2 < mdpStates; s2++){
    // Bellman Equation for (s, a, s2)
    float probability = mdp->getProbability(s, a, s2); // T(s, a, s')
    float reward = mdp->getReward(s, a, s2); // R(s, a, s')
    sum += probability * (reward + discount*f_prevStateValue[s2]);
  }
  return sum;
}

float MDPValueIterator::calcStateValue(int s){
  // create vector to hold all action sums
  if(mdpActions == 0) return 0.0;
  float maxQStateVal = calcQStateValue(s, 0); // (actions > 0)
  setQStateValue(s, 0, maxQStateVal); // update Q*k(s, a)
  for(int a = 1; a < mdpActions; a++){
    float currQStateVal = calcQStateValue(s, a);
    setQStateValue(s, a, currQStateVal); // update Q*k(s, a)
    if(currQStateVal > maxQStateVal){
      maxQStateVal = currQStateVal;
    }
  }
  return maxQStateVal;
}

void MDPValueIterator::vIterate(){
  k++;
  // copy Vk values to previous Vk
  f_prevStateValue.assign(f_stateValue.begin(), f_stateValue.end());
  // calculate V*(s) for each state in mdp
  for(int s = 0; s < mdpStates; s++){
    setStateValue(s, calcStateValue(s));
  }
}