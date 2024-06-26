import gym
import numpy as np
import utils
import matplotlib.pyplot as plt
import os


def sample(theta, env, N):
    """samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

    Note: the maximum trajectory length is 200 steps
    """
    total_rewards = []
    total_grads = []
    for n in range(N):
        trajectory_grads = []
        trajectory_rewards = []

        # TODO Get initial state
        observation = env.reset()
        for t in range(200):
            # TODO Extract features, get trajectory_grads and get trajectory_rewards
            phis = utils.extract_features(observation, env.action_space.n)

            # Get action distribution and sample action
            action_dist = utils.compute_action_distribution(theta, phis)
            action = np.random.choice(env.action_space.n, p=action_dist.squeeze())

            # Compute log softmax gradient
            log_softmax_grad = utils.compute_log_softmax_grad(theta, phis, action)

            # Step in the environment and store the gradient and reward
            observation, reward, done, _ = env.step(action)
            trajectory_grads.append(log_softmax_grad)
            trajectory_rewards.append(reward)

            # Terminate loop if done
            if done:
                break

        total_rewards.append(trajectory_rewards)
        total_grads.append(trajectory_grads)

    return total_grads, total_rewards


def train(N, T, delta, lamb=1e-3):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100, 1)
    env = gym.make("CartPole-v0")
    env.seed(12345)

    episode_rewards = []

    for t in range(T):
        print(f"Iteration {t}")
        # TODO Update theta according to handout, and record rewards
        # Sample trajectories
        trj_grads, trj_rewards = sample(theta, env, N)

        # Compute fisher matrix, value gradient and eta
        fisher = utils.compute_fisher_matrix(trj_grads, lamb)
        v_grad = utils.compute_value_gradient(trj_grads, trj_rewards)
        eta = utils.compute_eta(delta, fisher, v_grad)

        # Update theta
        theta += eta * np.linalg.inv(fisher) @ v_grad

        # Add average episode reward to list
        episode_rewards.append(np.mean([sum(trj) for trj in trj_rewards]))

    return theta, episode_rewards


if __name__ == "__main__":
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    theta_dir = "./learned_policies/NPG"
    os.makedirs(theta_dir, exist_ok=True)
    np.save(os.path.join(theta_dir, "expert_theta.npy"), theta)

    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plot_dir = "./plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "rewards"))
    plt.show()
