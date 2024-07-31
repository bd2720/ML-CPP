#include "mdp.h"
#include "mdpValueIterator.h"
using namespace std;

MDPValueIterator::MDPValueIterator(MDP * mdpRef, float discountRate){
  this->mdp = mdpRef;
  this->discount = discountRate;
  k = 0; // 0th iteration
  f_stateValue.resize(mdp->n_states);
  f_prevStateValue.resize(mdp->n_states);
}

float MDPValueIterator::getDiscount(){
  return this->discount;
}

int MDPValueIterator::getCurrentK(){
  return this->k;
}

float MDPValueIterator::calcQStateValue(int s, int a){
  float sum = 0.0;
  // sum together value over all destination states
  for(int s2 = 0; s2 < mdp->n_states; s2++){
    // Bellman Equation for (s, a, s2)
    float probability = mdp->getProbability(s, a, s2); // T(s, a, s')
    float reward = mdp->getReward(s, a, s2); // R(s, a, s')
    sum += probability * (reward + discount*f_prevStateValue[s2]);
  }
  return sum;
}

float MDPValueIterator::calcStateValue(int s){
  // create vector to hold all action sums
  if(mdp->n_actions == 0) return 0.0;
  float maxQStateVal = calcQStateValue(s, 0); // (actions > 0)
  for(int a = 1; a < mdp->n_actions; a++){
    float currQStateVal = calcQStateValue(s, a);
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
  for(int s = 0; s < mdp->n_states; s++){
    f_stateValue[s] = calcStateValue(s);
  }
}