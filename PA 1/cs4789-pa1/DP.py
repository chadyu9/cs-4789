import numpy as np


class DynamicProgramming:
    def __init__(self, MDP):
        self.R = MDP.R  # |A|x|S|
        self.P = MDP.P  # |A|x|S|x|S|
        self.horizon = MDP.horizon
        self.nStates = MDP.nStates
        self.nActions = MDP.nActions

    ####Helpers####
    def extractRpi(self, pi):
        """
        Returns R(s, pi_t(s)) for all timesteps t \in [H] and all states. Thus, the output will be an array of H x |S| entries.
        This should be used in policy evaluation and policy iteration.

        Parameter pi: a deterministic policy
        Precondition: An array of H x |S| integers, each of which specifies an action (row) for a given state s.

        HINT: Given an m x n matrix A, the expression

        A[row_indices, col_indices] (len(row_indices) == len(col_indices))

        returns a matrix of size len(row_indices) that contains the elements

        A[row_indices[i], col_indices[i]] in a row for all indices i.
        """
        temp_array = []
        for i in range(self.horizon):
            temp_array.append(self.R[pi[i].astype(int), np.arange(len(self.R[0]))])

        return np.stack(temp_array)

    def extractPpi(self, pi):
        """
        Returns P^pi_t for all timesteps t \in [H]: This is a H x |S|x|S| matrix where the (i,j) entry corresponds to
        P(j|i, pi_t(i))

        Parameter pi: a deterministic policy
        Precondition: An array of H x |S| integers
        """
        temp_array = []
        for i in range(self.horizon):
            temp_array.append(self.P[pi[i].astype(int), np.arange(len(self.P[0]))])

        return np.stack(temp_array)

    ####Dynamic Programming###
    def computeQfromV(self, V_next):
        """
        Returns the Q function given a V function for all actions at timestep t with known V_{t+1}.
        The output is an |S|x|A| array.
        You can assume that V is filled in the for the future timesteps.

        Use the bellman equation for Q-function to compute Q from V.

        Parameter V: value function
        Precondition: An array of |S| numbers
        """
        # TODO 1
        # Placeholder, replace with your code.
        return (self.R + np.dot(self.P, V_next)).T

    def dynamicProgramming(self):
        """
        This function runs on the from the last timestep to the first timestep and calculates
        {Q_t} for all t \in [H]. Hint: You probably will want to use self.horizon, self.R, and
        self.P in your solution.

        This function should return the policy and value function.
        """
        # TODO 2
        # Placeholder, replace with your code.
        # Initialize the policy and value function matrices
        pi = np.zeros((self.horizon, self.nStates))
        V = np.zeros((self.horizon, self.nStates))

        # Iterate backward through the timesteps
        for t in range(self.horizon - 1, -1, -1):
            # When t = H - 1, we need to compute Q_H from V_H = 0
            if t == self.horizon - 1:
                Q_t = self.computeQfromV(np.zeros(self.nStates))
            else:
                Q_t = self.computeQfromV(V[t + 1])

            # Fill in optimal policy and value function
            pi[t] = np.argmax(Q_t, axis=1)
            V[t] = Q_t[np.arange(self.nStates), pi[t].astype(np.int32)]

        return pi, V
