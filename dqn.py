import numpy as np


def dqn_run(env, agent, render=False, with_prints=False):
	state = env.reset()
	reward = 0
	for time_t in range(5501):
		if render:
			env.render()
		action = agent.act(state)	
		state_next, reward, done, info = env.step(action)
		agent.remember(state, action, reward, state_next, done)
		loss = agent.memory_replay(time_t)

		state = state_next
		reward += reward
		if done:
			state = env.reset()
		if not with_prints:
			continue
		if time_t % 1000 == 0:
			print('Reward : {}'.format(reward / 1000))
			print('Loss: {}'.format(loss))
			reward = 0

		print(f'Actions done: {time_t}', end='\r')	

	env.close()
	return agent	
