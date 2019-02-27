import torch
import numpy as np
import random
import copy
from model_vasya import ModelVasya
from collections import deque
from scipy.spatial import distance


class AgentPetya:
	def __init__(self, memory_size=30000, model_path='', batch_size=48, gamma=0.99, load=False, model_update_t=5000, 
											actions_num=6, epsilon=1, checkpoint_t=10000, model_eval=False):
		self.actions = self._discretize_actions(actions_num)
		self.memory = deque(maxlen=memory_size)
		self.batch_size = batch_size
		self.gamma = gamma
		self.epsilon = epsilon
		self.checkpoint_t = checkpoint_t
		self.model_update_t = model_update_t
		self.model_path = model_path
		self._set_model(actions_num, load, model_eval)


	def act(self, state):
		if np.random.rand() < self.epsilon:
			return np.array([np.random.choice(self.actions)])
		actions_values = self._predict(self.model, torch.from_numpy(state).float())
		action_index = np.argmax(actions_values)
		return np.array([self.actions[action_index]])


	def remember(self, state, action, reward, state_next, done):
		self.memory.append((state, action, reward, state_next, done))


	def memory_replay(self, time_t):
		if len(self.memory) < self.batch_size:
			return 0
		minibatch = np.array(random.sample(self.memory, self.batch_size))

		non_final_mask = np.where(minibatch[:, 4] == 0)
		batch_non_final_next_states = torch.tensor(list(minibatch[non_final_mask][:, 3]))
		batch_states = torch.tensor(list(minibatch[:, 0])).float()
		batch_actions_indexes = self._map_action_to_dscrt_act_ind(minibatch[:, 1])
		batch_rewards = torch.tensor(list(minibatch[:, 2]))

		next_state_values = np.zeros(self.batch_size)
		next_state_values[non_final_mask] = np.max(self._predict(self.hat_model, batch_non_final_next_states.float()).numpy(), axis=1)
		q_values_updated = next_state_values * self.gamma + batch_rewards
		
		q_values = self.model(batch_states).gather(1, torch.from_numpy(batch_actions_indexes))

		loss = self._update_model(q_values, q_values_updated.float().unsqueeze(1))

		if time_t % self.model_update_t == 0:
			self.hat_model = copy.deepcopy(self.model)
		if time_t % self.checkpoint_t == 0 and self.model_path:
			self._save_checkpoint(time_t)	
		if self.epsilon > 0.01:
			self.epsilon *= 0.999
		return loss


	def evaluate(self):
		self.model.eval()
		self.epsilon = 0


	def _set_model(self, output_dim, load_from_file=False, model_eval=False):
		self.model = ModelVasya(output_dim)
		self.optimizer = torch.optim.Adam(self.model.parameters())
		self.loss_fn = torch.nn.functional.mse_loss
		if load_from_file:
			checkpoint = torch.load(self.model_path)
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.loss_fn = checkpoint['loss']
			if model_eval:
				self.model.eval()
			else:
				self.model.train()
		self.hat_model = copy.deepcopy(self.model)


	def _save_checkpoint(self, epoch):
		torch.save({
			'epoch': epoch,
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss_fn
			}, self.model_path)
		return True


	def _predict(self, model, data_x):
		with torch.no_grad():
			return model(data_x)


	def _update_model(self, q_values, q_values_updated):
		loss = self.loss_fn(q_values, q_values_updated)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.data.item()


	def _discretize_actions(self, actions_num):
		high = 24
		low = -24
		actions = np.around(np.linspace(low, high, actions_num), decimals=3)
		# actions = np.array([-22, -18, -12, -9, -6, -4, -2, 2, 4, 6, 9, 12, 18, 22])
		return actions


	def _map_action_to_dscrt_act_ind(self, actions):
		dist = distance.cdist(np.expand_dims(self.actions, axis=1), np.expand_dims(actions, axis=1))
		actions_inds = np.argmin(dist, axis=0)
		return actions_inds.reshape(len(actions_inds), 1)
