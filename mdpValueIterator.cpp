#include "mdp.h"
#include "mdpValueIterator.h"
using namespace std;

MDPValueIterator::MDPValueIterator(MDP * mdpRef){
  this->mdp = mdpRef;
  k = 0; // 0th iteration
  f_stateValue.resize(mdp->n_states);
  f_prevStateValue.resize(mdp->n_states);
}

int MDPValueIterator::getCurrentK(){
  return this->k;
}

void MDPValueIterator::vIterate(){
  k++;
  // copy Vk values to previous Vk
  f_prevStateValue.assign(f_stateValue.begin(), f_stateValue.end());
  // calculate V*(s) for each state in mdp
  for(int s = 0; s < mdp->n_states; s++){
    
  }
}
