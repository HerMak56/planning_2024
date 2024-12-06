import numpy as np
from utils import *


class MDP:

    def __init__(self,
                 env: Environment,
                 goal: tuple,
                 gamma: float = 0.99):
        """
        env is the grid enviroment
        goal is the goal state
        gamma is the discount factor
        """
        self._env = env
        self._goal = goal
        self._gamma = gamma
        self._V = np.zeros(env.shape)
        self._policy = np.zeros(self._env.shape, 'b') #type byte (or numpy.int8)

    def calculate_value_function(self):
        """
        This function uses the Value Iteration algorithm to fill in the
        optimal value function
        """
        max_iters = 100
        for iter in range(max_iters):
            for i in range(self._env.shape[0]):
                for j in range(self._env.shape[1]):
                    state = (i, j)
                    if not self._env.state_consistency_check(state) or state == self._goal:
                        continue  # skip states with collision and goal state
                
                    max_value = float('-inf')
                    for action in action_space:
                        state_propagated_list, prob_list = self._env.probabilistic_transition_function(state, action)
                        value = 0

                        for next_state, prob in zip(state_propagated_list, prob_list):
                            if not self._env.state_consistency_check(next_state):
                                continue
                            reward = -1 if not self._env.state_consistency_check(next_state) else 0
                            if next_state == self._goal:
                                reward = 1
                            value += prob * (reward + self._gamma * self._V[next_state])

                        max_value = max(max_value, value)

                    self._V[state] = max_value
            
        return self._V

    def calculate_policy(self):
        """
        Only to be run AFTER Vopt has been calculated.
        
        output:
        policy: a map from each state s to the greedy best action a to execute
        """

        for i in range(self._env.shape[0]):
            for j in range(self._env.shape[1]):
                state = (i, j)
                if not self._env.state_consistency_check(state) or state == self._goal:
                    continue 

                best_action = None
                max_value = -np.inf

                for index,action in enumerate(action_space):
                    state_propagated_list, prob_list = self._env.probabilistic_transition_function(state, action)
                    value = 0

                    for next_state, prob in zip(state_propagated_list, prob_list):
                        if not self._env.state_consistency_check(next_state):
                            continue

                        reward = -1 if not self._env.state_consistency_check(next_state) else 0
                        if next_state == self._goal:
                            reward = 1
                        value += prob * (reward + self._gamma * self._V[next_state])

                    if value > max_value:
                        max_value = value
                        best_action = index

                self._policy[state] = best_action 

        return self._policy

    def policy(self,state:tuple) -> int:
        """
        returns the action according to the policy
        """
        return self._policy[state]

