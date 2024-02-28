import numpy as np
from pa2.local_controller import LocalController


class LQRController:
    def __init__(self, env):
        self.local_controller = LocalController(env)

    def compute_Q_params(self, A, B, Q, R, M, q, r, m, b, P, y, p):
        """
        Compute the Q function parameters for time step t.
        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
            Parameters:
            A (2d numpy array): A numpy array with shape (n_s, n_s)
            B (2d numpy array): A numpy array with shape (n_s, n_a)
            Q (2d numpy array): A numpy array with shape (n_s, n_s). Q is PD
            R (2d numpy array): A numpy array with shape (n_a, n_a). R is PD.
            M (2d numpy array): A numpy array with shape (n_s, n_a)
            q (2d numpy array): A numpy array with shape (n_s, 1)
            r (2d numpy array): A numpy array with shape (n_a, 1)
            m (2d numpy array): A numpy array with shape (n_s, 1)
            b (1d numpy array): A numpy array with shape (1,)
            P (2d numpy array): A numpy array with shape (n_s, n_s). This is the quadratic term of the
                value function equation from time step t+1. P is PSD.
            y (2d numpy array): A numpy array with shape (n_s, 1).  This is the linear term
                of the value function equation from time step t+1
            p (1d numpy array): A numpy array with shape (1,).  This is the constant term of the
                value function equation from time step t+1
        Returns:
            C (2d numpy array): A numpy array with shape (n_s, n_s)
            D (2d numpy array): A numpy array with shape (n_s, n_a)
            E (2d numpy array): A numpy array with shape (n_s, n_a)
            f (2d numpy array): A numpy array with shape (n_s,1)
            g (2d numpy array): A numpy array with shape (n_a,1)
            h (1d numpy array): A numpy array with shape (1,)

            where the following equation should hold
            Q_t^*(s) = s^T C s + a^T D s + s^T E a + f^T s  + g^T a + h

        """
        C = Q + (A.T @ P @ A)
        D = R + (B.T @ P @ B)
        E = M + (2 * (A.T @ P @ B))
        ft = q.T + (2 * (m.T @ P @ A)) + (y.T @ A)
        gt = r.T + (2 * (m.T @ P @ B)) + (y.T @ B)
        h = (m.T @ P @ m) + (y.T @ m + p) + b

        return C, D, E, ft.T, gt.T, h.flatten()

    def compute_policy(self, A, B, m, C, D, E, f, g, h):
        """
        Compute the optimal policy at the current time step t
        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)


        Let Q_t^*(s) = s^T C s + a^T D a + s^T E a + f^T s  + g^T a  + h
        Parameters:
            A (2d numpy array): A numpy array with shape (n_s, n_s)
            B (2d numpy array): A numpy array with shape (n_s, n_a)
            m (2d numpy array): A numpy array with shape (n_s, 1)
            C (2d numpy array): A numpy array with shape (n_s, n_s). C is PD.
            D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
            E (2d numpy array): A numpy array with shape (n_s, n_a)
            f (2d numpy array): A numpy array with shape (n_s, 1)
            g (2d numpy array): A numpy array with shape (n_a, 1)
            h (1d numpy array): A numpy array with shape (1, )
        Returns:
            K_t (2d numpy array): A numpy array with shape (n_a, n_s)
            k_t (2d numpy array): A numpy array with shape (n_a, 1)

            where the following holds
            \pi*_t(s) = K_t s + k_t
        """
        D_inv = np.linalg.inv(D)
        K = -0.5 * (D_inv @ E.T)
        k = -0.5 * (D_inv @ g)
        return K, k.flatten()

    def compute_V_params(self, A, B, m, C, D, E, f, g, h, K, k):
        """
        Compute the V function parameters for the next time step
        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
        Let V_t^*(s) = s^TP_ts + y_t^Ts + p_t
        Parameters:
            A (2d numpy array): A numpy array with shape (n_s, n_s)
            B (2d numpy array): A numpy array with shape (n_s, n_a)
            m (2d numpy array): A numpy array with shape (n_s, 1)
            C (2d numpy array): A numpy array with shape (n_s, n_s). C is PD.
            D (2d numpy array): A numpy array with shape (n_a, n_a). D is PD.
            E (2d numpy array): A numpy array with shape (n_s, n_a)
            f (2d numpy array): A numpy array with shape (n_s, 1)
            g (2d numpy array): A numpy array with shape (n_a, 1)
            h (1d numpy array): A numpy array with shape (1, )
            K (2d numpy array): A numpy array with shape (n_a, n_s)
            k (2d numpy array): A numpy array with shape (n_a, 1)

        Returns:
            P_t (2d numpy array): A numpy array with shape (n_s, n_s)
            y_t (2d numpy array): A numpy array with shape (n_s, 1)
            p_t (1d numpy array): A numpy array with shape (1,)
        """
        P = C + (K.T @ D @ K) + (E @ K)
        yt = (2 * (k.T @ D @ K)) + (k.T @ E.T) + (g.T @ K) + f.T
        p = (k.T @ D @ k) + (g.T @ k) + h

        return P, yt.T, p.flatten()

    def lqr(self, s_star, a_star, T):
        """
        Compute optimal policies by solving
        argmin_{\pi_0,...\pi_{T-1}} \sum_{t=0}^{T-1} s_t^T Q s_t + a_t^T R a_t + s_t^T M a_t + q^T s_t + r^T a_t
        subject to s_{t+1} = A s_t + B a_t + m, a_t = \pi_t(s_t)

        Let the shape of s_t be (n_s,), the shape of a_t be (n_a,)
        Let optimal \pi*_t(s) = K_t s + k_t

        Parameters:
        s_star (numpy array) with shape (4,)
        a_star (numpy array) with shape (1,)
        T (int): The number of total steps in finite horizon settings

        Returns:
            ret (list): A list, [(K_0, k_0), (K_1, k_1), ..., (K_{T-1}, k_{T-1})]
            and the shape of K_t is (n_a, n_s), the shape of k_t is (n_a,)
        """
        # TODO
        N_s = s_star.shape[0]
        N_a = a_star.shape[0]

        # get A, B, Q, R, M, q, r
        A, B, Q, R, M, q, r = self.local_controller.compute_local_expansions(
            s_star, a_star
        )

        # Create H block and make PD
        H = np.concatenate((np.concatenate((Q, M.T)), np.concatenate((M, R))), axis=1)
        # When there is an eigenvalue that is not positive, force H to be PD
        if not np.all(np.linalg.eigvals(H) > 0):
            H_eval, H_evec = np.linalg.eig(H)
            H = sum(
                [
                    (
                        H_eval[i] * np.outer(H_evec[i], H_evec[i])
                        if H_eval[i] > 0
                        else np.zeros((N_s + N_a, N_s + N_a))
                    )
                    + 1e-5 * np.eye(N_s + N_a)
                    for i in range(N_s + N_a)
                ]
            )

        # Extract updated Q_2, M, R_2, q_2, r_2, b, m
        Q, R, M = H[:N_s, :N_s], H[N_s:, N_s:], H[:N_s, N_s:]
        Q_2, R_2, q_2, r_2, b, m = (
            Q / 2,
            R / 2,
            (q.T - s_star.T @ Q - a_star.T @ M.T).T,
            (r.T - a_star.T @ R - s_star.T @ M).T,
            self.local_controller.c(s_star, a_star)
            + 0.5 * (s_star.T @ Q / 2 @ s_star + a_star.T @ R @ a_star)
            + s_star.T @ M @ a_star
            - q.T @ s_star
            - r.T @ a_star,
            (self.local_controller.f(s_star, a_star) - A @ s_star - B @ a_star).reshape(
                -1, 1
            ),
        )

        # Compute K, k with base step time t = T-1
        policy = [(-0.5 * np.linalg.solve(R_2, M.T), -0.5 * np.linalg.solve(R_2, r_2))]

        # Compute parameters of V_{T-1}^{star}
        (
            P,
            y,
            p,
        ) = (
            Q_2 - 0.25 * M @ np.linalg.solve(R_2, M.T),
            (q_2.T - 0.5 * r.T @ np.linalg.solve(R_2, M.T)).T,
            (b - 0.25 * r.T @ np.linalg.solve(R_2, r_2)).flatten(),
        )

        # Loop through other time steps inductively
        for _ in range(T - 2, -1, -1):
            # Compute parameters of Q_t^{star}
            C, D, E, f, g, h = self.compute_Q_params(
                A, B, Q_2, R_2, M, q_2, r_2, m, b, P, y, p
            )

            # Compute K_t, k_t and add it to policy list
            K_t, k_t = self.compute_policy(D=D, E=E, g=g)
            policy = [(K_t, k_t)] + policy

            # Compute parameters of V_t^{star}
            P, y, p = self.compute_V_params(C=C, D=D, E=E, f=f, g=g, h=h, K=K_t, k=k_t)

        # return policy
        return policy
