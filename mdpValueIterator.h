#ifndef MDP_VALUE_ITERATOR_H
#define MDP_VALUE_ITERATOR_H
#include "mdp.h"

class MDPValueIterator {
  private:
    MDP * mdp;      // reference to MDP describing the problem
    int mdpStates;  // number of states in MDP
    int mdpActions; // number of actions in MDP

    float discount; // discount rate, in (0.0, 1.0)
    int k; // current iteration (k=0 --> v*(s) = 0.0)
    vector<float> f_newStateValue;       // V*k+1(s) temp space

    vector<float> f_stateValue;       // V*k(s)     --> utility
    vector<float> f_qStateValue;      // Q*k(s, a)  --> utility
    vector<float> f_extractedPolicy;  // pi_k(s)    --> action

    // index into f_qStateValue given (s, a)
    int q_idx(int s, int a);

    // calculate Q*k(s, a) for a given q-state
    float calcQStateValue(int s, int a);
    // calculate V*k(s) for a given state "s"
    float calcStateValue(int s);
    
    // set value of qState in f_qStateValue using q_idx
    void setQStateValue(int s, int a, float val);
    // set policy value in state s to bestAction
    void setExtractedPolicy(int s, int bestAction);

  public:
    // create a Value Iterator to solve the given MDP
    MDPValueIterator(MDP * mdpRef, float discountRate=1.0);
    // get discount value
    float getDiscount(void);
    // return current iteration number
    int getCurrentK(void);

    // return q-state value (after already calculated)
    float getQStateValue(int s, int a);
    // return state value (after already calculated)
    float getStateValue(int s);
    // return best action in state s (after already calculated)
    int getExtractedPolicy(int s);
    // calculate the next iteration of V*k(s), Q*k(s, a)
    void vIterate(void);
    // calculate pi_k(s) for each state
    void extractPolicy(void);
};

#endif