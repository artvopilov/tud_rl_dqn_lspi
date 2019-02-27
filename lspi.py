import numpy as np

def lspi_run(env,agent):
    best_mean_rewards = 0
    rewards = []
    for i in range(5000):
        rewards.append(play(env, agent))

        if i %10==0:
            print('Iteration:',i,
            'mean reward of last 10 episodes =', np.mean(rewards[-10:]),
            'best reward =', best_mean_rewards)

            if np.mean(rewards[-10:]) > best_mean_rewards:
                best_w = agent.w
                best_mean_rewards = np.mean(rewards[-10:])

            s,a,r,next_s = agent.sampling(env,200)
            agent.LSPI(s,a,r,next_s)

        if best_mean_rewards >= 19999.5:
            agent.w = best_w
            break

    return agent

def play(env,agent,t_max=20000):
    total_reward = 0.0
    s = env.reset()

    for t in range(t_max):
        a = agent.play(s)
        next_s, r, done, _ = env.step(a)

        s = next_s
        total_reward +=r

        if done:
            break

    return total_reward
