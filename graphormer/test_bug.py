import torch
import torch.nn as nn

pos_size=23
tim_dim=32
hidden_dim=512

data=[1,2,3,4,5]

time_encoder = nn.Embedding(169, tim_dim, padding_idx=0)
pos_embedding = nn.Embedding(pos_size,hidden_dim)
node_tim_cat = nn.Linear(hidden_dim+tim_dim, hidden_dim)
# node_tim_cat = nn.Linear(hidden_dim, hidden_dim)