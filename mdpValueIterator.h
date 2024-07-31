#ifndef MDP_VALUE_ITERATOR_H
#define MDP_VALUE_ITERATOR_H
#include "mdp.h"

class MDPValueIterator {
  private:
    MDP * mdp; // reference to MDP describing the problem
    float discount; // discount rate, in (0.0, 1.0)
    int k; // current iteration (k=0 --> v*(s) = 0.0)
    vector<float> f_prevStateValue; // V*k-1(s)

    // calculate Q*k(s, a) for a given q-state
    float calcQStateValue(int s, int a);
    // calculate V*k(s) for a given state "s"
    float calcStateValue(int s);

  public:
    vector<float> f_stateValue; // V*k(s)
    
    // create a Value Iterator to solve the given MDP
    MDPValueIterator(MDP * mdpRef, float discountRate);
    // get discount value
    float getDiscount(void);
    // return current iteration number
    int getCurrentK(void);
    // calculate the next iteration of V*k()
    void vIterate(void);
};

#endif