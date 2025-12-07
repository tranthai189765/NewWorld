from abc import ABC 
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
        # nn.init.kaiming_normal_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module, ABC):
    def __init__(self, n_states, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_states = n_states
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(self.n_states, self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()

        self.hidden2 = nn.Linear(self.n_hidden_filters, self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.q = nn.Linear(self.n_hidden_filters, self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()
    
    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.q(x)
        return logits
    
class ValueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_hidden_filters=256):
        super(ValueNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(self.n_states, self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()

        self.hidden2 = nn.Linear(self.n_hidden_filters, self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.value = nn.Linear(self.n_hidden_filters, 1)
        init_weight(self.value, initializer="xavier uniform")
        self.value.bias.data.zero_()
    
    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        value = self.value(x)
        return value

class QvalueNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(QvalueNetwork, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(self.n_states, self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        
        self.hidden2 = nn.Linear(self.n_hidden_filters, self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.q_value = nn.Linear(self.n_hidden_filters, self.n_actions)
        init_weight(self.q_value, initializer="xavier uniform")
        self.q_value.bias.data.zero_()
    
    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        values = self.q_value(x)
        return values

class DiscretePolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden_filters=256):
        super(DiscretePolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden_filters = n_hidden_filters

        self.hidden1 = nn.Linear(self.n_states, self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()

        self.hidden2 = nn.Linear(self.n_hidden_filters, self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.logits = nn.Linear(self.n_hidden_filters, self.n_actions)
        init_weight(self.logits, initializer="xavier uniform")
        self.logits.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))
        logits = self.logits(x)
        return logits
    
    def sample_or_likelihood(self, states):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        #unsqueeze(-1): them vao chieu cuoi cung
        return action, log_prob

class PolicyNetwork(nn.Module, ABC):
    def __init__(self, n_states, n_actions, action_bounds, n_hidden_filters=256):
        super(PolicyNetwork, self).__init__()
        self.n_states = n_states
        self.n_hidden_filters = n_hidden_filters
        self.n_actions = n_actions
        self.action_bounds = action_bounds

        self.hidden1 = nn.Linear(in_features=self.n_states, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()

        self.mu = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.mu, initializer="xavier uniform")
        self.mu.bias.data.zero_()

        self.log_std = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_actions)
        init_weight(self.log_std, initializer="xavier uniform")
        self.log_std.bias.data.zero_()

    def forward(self, states):
        x = F.relu(self.hidden1(states))
        x = F.relu(self.hidden2(x))

        mu = self.mu(x)
        log_std = self.log_std(x)
        std = log_std.clamp(min=-20, max=2).exp()
        dist = Normal(mu, std)
        return dist

    def sample_or_likelihood(self, states):
        dist = self(states)
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)
        # Enforcing action bounds
        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return (action * self.action_bounds[1]).clamp_(self.action_bounds[0], self.action_bounds[1]), log_prob
    