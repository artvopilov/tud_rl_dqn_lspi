import torch
from torch import nn


class ModelVasya(nn.Module):

	def __init__(self, output_dim):
		super(ModelVasya, self).__init__()
		self.linear1 = nn.Linear(5, 12)
		self.layer_norm1 = nn.LayerNorm(12)
		self.linear2 = nn.Linear(12, 24)
		self.layer_norm2 = nn.LayerNorm(24)
		self.linear3 = nn.Linear(24, 16)
		self.layer_norm3 = nn.LayerNorm(16)
		self.head = nn.Linear(16, output_dim)

	def forward(self, x):
		x = torch.tanh(self.linear1(x))
		x = torch.tanh(self.linear2(x))
		x = torch.tanh(self.linear3(x))
		return self.head(x)
