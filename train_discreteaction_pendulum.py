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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha = 1e-4
    gamma = 0.95
    batch_size = 128
    num_episodes = 1000
    reset_num = 100

    r_list = []
    dr_list = []
    dr_list2 = []
    dr_list3 = []
    dr_list4 = []

    for i in range(5):
        print('set 1',i)
        agent = DQN.Agent(env.num_states, env.num_actions, alpha, gamma, batch_size, replay=True)
        rewards, disc_rewards = agent.train(num_episodes, env, reset_num, targetQ=True)
        r_list.append(rewards)
        dr_list.append(disc_rewards)

        _, disc_rewards = agent.train(num_episodes, env, reset_num, targetQ=False)
        dr_list2.append(disc_rewards)

    for i in range(5):
        print('set 2',i)
        agent = DQN.Agent(env.num_states, env.num_actions, alpha, gamma, batch_size, replay=False)
        _, disc_rewards = agent.train(num_episodes, env, reset_num, targetQ=True)
        dr_list3.append(disc_rewards)

        _, disc_rewards = agent.train(num_episodes, env, reset_num, targetQ=False)
        dr_list4.append(disc_rewards)

    r_mean = np.array(r_list).mean(axis=0)
    r_std = np.array(r_list).std(axis=0)

    dr_mean = np.array(dr_list).mean(axis=0)
    dr_std = np.array(dr_list).std(axis=0)

    dr_mean2 = np.array(dr_list2).mean(axis=0)
    dr_std2 = np.array(dr_list2).std(axis=0)

    dr_mean3 = np.array(dr_list3).mean(axis=0)
    dr_std3 = np.array(dr_list3).std(axis=0)

    dr_mean4 = np.array(dr_list4).mean(axis=0)
    dr_std4 = np.array(dr_list4).std(axis=0)

    plt.figure()
    x = np.arange(len(r_mean))
    plt.plot(x, r_mean, 'b-', label = 'undiscounted reward')
    plt.fill_between(x, r_mean - r_std, r_mean + r_std, color='b', alpha=0.2)
    plt.plot(x, dr_mean, 'r-', label = 'discounted reward')
    plt.fill_between(x, dr_mean - dr_std, dr_mean + dr_std, color='r', alpha=0.2)
    plt.xlabel('Episodes')
    plt.ylim((0,40))
    plt.legend()
    plt.savefig('figures/learning_curve.png')

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(rewards, label='undiscounted reward')
    ax[0].set_xlabel('Epiosdes')
    ax[0].set_ylim(bottom=0)
    ax[0].legend()
    ax[1].plot(disc_rewards, label='discounted reward')
    ax[1].set_xlabel('Epiosdes')
    ax[1].set_ylim(bottom=0)
    ax[1].legend()
    plt.savefig('figures/learning_curve.png')

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

    # ablation study
    # with replay & target (standard algorithm)
    # with replay & without target Q
    # without replay, with target Q
    # without replay, without target Q

    plt.figure()
    x = np.arange(len(r_mean))
    plt.plot(x, dr_mean, 'r-', label='reward (with replay, with target Q)')
    plt.fill_between(x, dr_mean - dr_std, dr_mean + dr_std, color='r', alpha=0.2)
    plt.plot(x, dr_mean2, 'g-', label='reward (with replay, without target Q)')
    plt.fill_between(x, dr_mean2 - dr_std2, dr_mean2 + dr_std2, color='g', alpha=0.2)
    plt.plot(x, dr_mean3, 'b-', label='reward (without replay, with target Q)')
    plt.fill_between(x, dr_mean3 - dr_std3, dr_mean3 + dr_std3, color='b', alpha=0.2)
    plt.plot(x, dr_mean4, 'm-', label='reward (without replay, without target Q)')
    plt.fill_between(x, dr_mean4 - dr_std4, dr_mean4 + dr_std4, color='m', alpha=0.2)
    plt.xlabel('Episodes')
    plt.ylim((0,30))
    plt.legend()
    plt.savefig('figures/ablation_study.png')


if __name__ == '__main__':
    main()