import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

torch.manual_seed(0) # set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size=6, h_size=32, a_size=3, hidden_dim=[32], acti_fun = nn.ReLU()):
        super(Policy, self).__init__()
        #self.fc1 = nn.Linear(s_size, h_size)
        #self.fc2 = nn.Linear(h_size, a_size)
        self.layers = nn.ModuleList()
        self.acti = acti_fun
        current_dim = s_size
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, a_size))

    def forward(self, x):
        # = F.relu(self.fc1(x))
        #x = self.fc2(x)
        for layer in self.layers[:-1]:
            x = self.acti(layer(x))
        out = self.layers[-1](x)
        #return out
        return F.softmax(out, dim=1)
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        #action = torch.argmax(probs)
        return action.item() - 1, m.log_prob(action)
        #return action.item()-1, probs.numpy()[action.item()-1]