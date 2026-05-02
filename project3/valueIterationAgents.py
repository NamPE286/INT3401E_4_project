# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for _ in range(self.iterations):
            # We use a new counter for V_{k+1} to ensure we do batch updates.
            # We only use V_k (self.values) to compute these new values.
            new_values = util.Counter()
            
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    # Get the best action for the current state using V_k
                    best_action = self.computeActionFromValues(state)
                    if best_action is not None:
                        # Update V_{k+1}(s) to be the max Q-value
                        new_values[state] = self.computeQValueFromValues(state, best_action)
            
            # After finishing the sweep over all states, update the main values dict
            self.values = new_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        
        # transition is a tuple of (nextState, prob)
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            
            # Q(s,a) = sum_{s'} T(s,a,s') * [R(s,a,s') + discount * V(s')]
            q_value += prob * (reward + self.discount * self.values[nextState])
            
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Return None for terminal states
        if self.mdp.isTerminal(state):
            return None
            
        possible_actions = self.mdp.getPossibleActions(state)
        
        # Return None if no legal actions exist
        if not possible_actions:
            return None
            
        best_action = None
        max_q_value = float('-inf')
        
        # Find the action that yields the highest Q-value
        for action in possible_actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action
                
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        """
          Run the prioritized sweeping value iteration algorithm for the 
          specified number of iterations.
        """
        # 1. Compute predecessors of all states.
        # A predecessor is a state that can reach another state with probability > 0.
        predecessors = {}
        for state in self.mdp.getStates():
            predecessors[state] = set()

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)

        # 2. Initialize an empty priority queue.
        pq = util.PriorityQueue()

        # 3. For each non-terminal state s, do:
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                # Find the highest Q-value across all possible actions from s
                max_q = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                
                # Find the absolute value of the difference
                diff = abs(self.values[state] - max_q)
                
                # Push s into the priority queue with priority -diff
                pq.push(state, -diff)

        # 4. For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for _ in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if pq.isEmpty():
                break

            # Pop a state s off the priority queue.
            state = pq.pop()

            # Update the value of s (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(state):
                max_q = max([self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state)])
                self.values[state] = max_q

            # For each predecessor p of s, do:
            for p in predecessors[state]:
                if not self.mdp.isTerminal(p):
                    # Find the highest Q-value across all possible actions from p
                    max_q_p = max([self.computeQValueFromValues(p, action) for action in self.mdp.getPossibleActions(p)])
                    
                    # Find the absolute value of the difference
                    diff = abs(self.values[p] - max_q_p)

                    # If diff > theta, push p into the priority queue
                    if diff > self.theta:
                        pq.update(p, -diff)

