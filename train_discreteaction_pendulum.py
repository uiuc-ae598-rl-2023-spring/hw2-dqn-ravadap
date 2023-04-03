import DQN
import discreteaction_pendulum
import torch
import itertools
import matplotlib.pyplot as plt
import random
import numpy as np
import seaborn as sns

def main():
    env = discreteaction_pendulum.Pendulum()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha = 1e-4
    gamma = 0.95
    batch_size = 128
    num_episodes = 1000
    reset_num = 100

    r_list = []
    dr_list = []

    for i in range(5):
        agent = DQN.Agent(env.num_states, env.num_actions, alpha, gamma, batch_size)
        rewards, disc_rewards = agent.train(num_episodes, env, reset_num)
        r_list.append(rewards)
        dr_list.append(disc_rewards)

    print('here')
    r_mean = np.array(r_list).mean(axis=0)
    r_std = np.array(r_list).std(axis=0)

    print('here')
    dr_mean = np.array(dr_list).mean(axis=0)
    dr_std = np.array(dr_list).std(axis=0)

    # r_mean = np.array([10, 20, 30, 25, 32, 43])
    # r_std = np.array([2.2, 2.3, 1.2, 2.2, 1.8, 3.5])

    # dr_mean = np.array([12, 22, 30, 13, 33, 39])
    # dr_std = np.array([2.4, 1.3, 2.2, 1.2, 1.9, 3.5])
    print(len(r_mean), len(r_std), len(dr_mean), len(dr_std))

    plt.figure()
    x = np.arange(len(r_mean))
    plt.plot(x, r_mean, 'b-', label='discounted reward')
    plt.fill_between(x, r_mean - r_std, r_mean + r_std, color='b', alpha=0.2)
    plt.plot(x, dr_mean, 'r-', label='undiscounted reward')
    plt.fill_between(x, dr_mean - dr_std, dr_mean + dr_std, color='r', alpha=0.2)
    plt.xlabel('Episodes')
    plt.ylim((0,40))
    plt.legend()
    plt.savefig('figures/learning_curve.png')

    # fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # ax[0].plot(rewards, label='undiscounted reward')
    # ax[0].set_xlabel('Epiosdes')
    # ax[0].set_ylim(bottom=0)
    # ax[0].legend()
    # ax[1].plot(disc_rewards, label='discounted reward')
    # ax[1].set_xlabel('Epiosdes')
    # ax[1].set_ylim(bottom=0)
    # ax[1].legend()
    # plt.savefig('figures/learning_curve.png')

    # create video
    # Define a policy that maps every state to the "zero torque" action
    policy = lambda s: (agent.choose_action(s)).item()

    # Simulate an episode and save the result as an animated gif
    env.video(policy, filename='figures/test_discreteaction_pendulum.gif')

    # trajectory of agent
    # Initialize simulation
    s = env.reset()

    # Create dict to store data from simulation
    data = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = random.randrange(env.num_actions)
        (s, r, done) = env.step(a)
        data['t'].append(data['t'][-1] + 1)
        data['s'].append(s)
        data['a'].append(a)
        data['r'].append(r)

    # Parse data from simulation
    data['s'] = np.array(data['s'])
    theta = data['s'][:, 0]
    thetadot = data['s'][:, 1]
    tau = [env._a_to_u(a) for a in data['a']]

    # Plot data and save to png file
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(data['t'], theta, label='theta')
    ax[0].plot(data['t'], thetadot, label='thetadot')
    ax[0].legend()
    ax[1].plot(data['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(data['t'][:-1], data['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('figures/test_discreteaction_pendulum.png')

    # policy
    num = 100
    max_theta = np.pi
    max_thetadot = env.max_thetadot
    theta_vals = np.linspace(-max_theta, max_theta, num)
    thetadot_vals = np.linspace(-max_thetadot, max_thetadot, num)
    xv, yv = np.meshgrid(theta_vals, thetadot_vals)
    policy_grid = np.zeros((num, num))
    value_grid = np.zeros((num, num))

    for i in range(num):
        for j in range(num):
            s = np.array((xv[i,j], yv[i,j]))
            policy_grid[i,j] = policy(s)
            s_tensor = torch.tensor(s, dtype=torch.float).unsqueeze(0)
            value_grid[i,j] = agent.local_net(s_tensor).max().item()

    plt.figure(4)
    plt.pcolor(xv, yv, policy_grid)
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.colorbar()
    plt.savefig('figures/policy.png')
            
    plt.figure(5)
    plt.pcolor(xv, yv, value_grid)
    plt.xlabel('theta')
    plt.ylabel('thetadot')
    plt.colorbar()
    plt.savefig('figures/state-value.png')



if __name__ == '__main__':
    main()