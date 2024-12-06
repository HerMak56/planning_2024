import numpy as np
from utils import *


class VI:
    def __init__(self,
                 env: Environment,
                 goal: tuple):
        """
        env is the grid enviroment, as defined in utils
        goal is the goal state
        """
        self._env = env
        self._goal = goal
        self._G = np.ones(self._env.shape)*1e2
        # self._G = np.ones(self._env.shape)* np.inf
        self._policy = np.zeros(self._env.shape, 'b') #type byte (or numpy.int8)


    def calculate_value_function(self):
        """
        env is the grid enviroment
        goal is the goal state

        output:
        G: Optimal cost-to-go
        """
        max_iters = 100

        self._G[self._goal] = 0

        for iter in range(max_iters):
            for i in range(self._env.shape[0]):
                for j in range(self._env.shape[1]):

                    state = (i,j)
                    if not self._env.state_consistency_check(state) or state == self._goal:
                        continue # skip states with collision and goal state
                    costs_for_state = []
                    for action in action_space:
                        new_state, status = self._env.transition_function(s=state,a=action)
                        if status:
                            costs_for_state.append(1 + self._G[new_state])

                    if costs_for_state:
                        self._G[state] = min(costs_for_state)

        return self._G
        
    def calculate_policy(self):
        """
        G: optimal cot-to-go function (needed to be calcualte in advance)
        
        output:
        policy: a map from each state x to the best action a to execcute
        """
        for i in range(self._env.shape[0]):
            for j in range(self._env.shape[1]):
                state = (i,j)
                if not self._env.state_consistency_check(state) or state == self._goal:
                        continue # skip states with collision and goal state
                optimal_action = None
                min_cost = np.inf
                for index, action in enumerate(action_space):
                        new_state, status = self._env.transition_function(s=state,a=action)
                        if status:
                            cost = 1+ self._G[new_state]
                            if cost < min_cost:
                                min_cost = cost
                                optimal_action = index
                if optimal_action is not None:
                    self._policy[state] = optimal_action
        return self._policy
        
    def policy(self,state:tuple) -> int:
        """
        returns the action according to the policy
        """
        return self._policy[state]

