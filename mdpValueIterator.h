#ifndef MDP_VALUE_ITERATOR_H
#define MDP_VALUE_ITERATOR_H
#include "mdp.h"

class MDPValueIterator {
  private:
    MDP * mdp; // reference to MDP describing the problem
    int k; // current iteration (k=0 --> v*(s) = 0.0)
    vector<float> f_prevStateValue; // V*k-1(s)

  public:
    vector<float> f_stateValue; // V*k(s)
    
    // 
    MDPValueIterator(MDP * mdpRef);
    int getCurrentK(void);
    void vIterate(void);
};

#endif