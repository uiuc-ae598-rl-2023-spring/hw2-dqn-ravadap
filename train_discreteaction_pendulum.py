import DQN
import discreteaction_pendulum
import torch
import itertools
import matplotlib.pyplot as plt
import random
import numpy as np

def main():
    env = discreteaction_pendulum.Pendulum()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha = 1e-4
    gamma = 0.95
    batch_size = 128
    num_episodes = 100
    reset_num = 10

    agent = DQN.Agent(env.num_states, env.num_actions, alpha, gamma, batch_size)
    rewards = agent.train(num_episodes, env, reset_num)

    # rewards = []

    # for ep in range(num_episodes):
    #     print(ep)
    #     state = env.reset()
    #     state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
    #     reward = 0

    #     for t in itertools.count():
    #         # print(ep, t)
    #         action = agent.choose_action(state)

    #         next_state, reward_t, done = env.step(action.item())
    #         next_state = torch.tensor(next_state, dtype=torch.float, device=device).unsqueeze(0)
    #         reward_t = torch.tensor([reward_t], dtype=torch.float, device=device).unsqueeze(0)
    #         reward += reward_t.item()
            
    #         agent.memory.add(state, action, next_state, reward_t)
    #         state = next_state

    #         agent.optimize()

    #         if done: break

    #     rewards.append(reward)

    #     if ep % reset_num == 0:
    #         agent.target_net.load_state_dict(agent.local_net.state_dict())

    plt.figure(1)
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.ylim(bottom=0)
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
            value_grid[i,j] = np.max(policy(s))

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