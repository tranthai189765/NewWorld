import numpy as np
from .model import QvalueNetwork, ValueNetwork, Discriminator, DiscretePolicyNetwork
import torch
from .replay_memory import Memory, Transition
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn.functional import log_softmax
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class SACAgentDiscrete:
    def __init__(self,
                 p_z,
                 **config):
        
        self.config = config
        self.n_states = self.config["n_states"]
        self.n_local_states = self.config["n_local_states"]
        
        
        self.n_skills = self.config["n_skills"]
        self.batch_size = self.config["batch_size"]
        self.p_z = np.tile(p_z, self.batch_size).reshape(self.batch_size, self.n_skills)
        self.memory = Memory(self.config["mem_size"], self.config["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(self.config["seed"])
        self.policy_network = DiscretePolicyNetwork(n_states=self.n_states + self.n_skills,
                                            n_actions=self.config["n_actions"],
                                            n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network1 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.q_value_network2 = QvalueNetwork(n_states=self.n_states + self.n_skills,
                                              n_actions=self.config["n_actions"],
                                              n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                          n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.value_target_network = ValueNetwork(n_states=self.n_states + self.n_skills,
                                                 n_hidden_filters=self.config["n_hiddens"]).to(self.device)
        self.hard_update_target_network()

        self.discriminator = Discriminator(n_states=self.n_local_states, n_skills=self.n_skills,
                                           n_hidden_filters=self.config["n_hiddens"]).to(self.device)

        self.mse_loss = torch.nn.MSELoss()
        self.cross_ent_loss = torch.nn.CrossEntropyLoss()

        self.value_opt = Adam(self.value_network.parameters(), lr=self.config["lr"])
        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.config["lr"])
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.config["lr"])
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.config["lr"])
        self.discriminator_opt = Adam(self.discriminator.parameters(), lr=self.config["lr"])

    def choose_action(self, states):
        states = np.expand_dims(states, axis=0)
        states = from_numpy(states).float().to(self.device)
        action, _ = self.policy_network.sample_or_likelihood(states)
        return action.detach().cpu().numpy()[0]

    def store(self, state, z, done, action, next_state):
        state = from_numpy(state).float().to("cpu")
        z = torch.ByteTensor([z]).to("cpu")
        done = torch.BoolTensor([done]).to("cpu")
        action = torch.Tensor([action]).to("cpu")
        next_state = from_numpy(next_state).float().to("cpu")
        self.memory.add(state, z, done, action, next_state)

    def unpack(self, batch):
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)
        zs = torch.cat(batch.z).view(self.batch_size, 1).long().to(self.device)
        dones = torch.cat(batch.done).view(self.batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, 1).long().to(self.device)
        next_states = torch.cat(batch.next_state).view(self.batch_size, self.n_states + self.n_skills).to(self.device)

        return states, zs, dones, actions, next_states

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        else:
            batch = self.memory.sample(self.batch_size)
            states, zs, dones, actions, next_states = self.unpack(batch)
            p_z = from_numpy(self.p_z).to(self.device)

            # Calculating the value target
            reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states)
            q1_pi = self.q_value_network1(states) 
            q2_pi = self.q_value_network2(states)
            q1_selected = q1_pi.gather(1, reparam_actions.unsqueeze(-1))
            q2_selected = q2_pi.gather(1, reparam_actions.unsqueeze(-1))
            q = torch.min(q1_selected, q2_selected)
            target_value = q.detach() - self.config["alpha"] * log_probs.detach()

            value = self.value_network(states)
            value_loss = self.mse_loss(value, target_value)

            obs_next = torch.split(next_states, [self.n_states, self.n_skills], dim=-1)[0]
            obs_next_states = obs_next[:, 13:22]
            logits_next = self.discriminator(obs_next_states)
            
            p_z = p_z.gather(-1, zs)
            logq_z_ns = log_softmax(logits_next, dim=-1)
            rewards = logq_z_ns.gather(-1, zs).detach() - torch.log(p_z + 1e-6)

            # Calculating the Q-Value target
            with torch.no_grad():
                target_q = self.config["reward_scale"] * rewards.float() + \
                           self.config["gamma"] * self.value_target_network(next_states) * (~dones)


            actions_idx = actions.long()

            q1_all = self.q_value_network1(states) 
            q2_all = self.q_value_network2(states)
            q1_selected_actions = q1_all.gather(1, actions_idx)
            q2_selected_actions = q2_all.gather(1, actions_idx)
            q1_loss = self.mse_loss(q1_selected_actions, target_q)
            q2_loss = self.mse_loss(q2_selected_actions, target_q)

            policy_loss = (self.config["alpha"] * log_probs - q).mean()
            curr_obs = torch.split(states, [self.n_states, self.n_skills], dim=-1)[0]
            # print("obs = ", obs[4])
            obs_states = curr_obs[:, 13:22]
            # print("obs_states = ", obs_states)
            # time.sleep(1000)
            logits = self.discriminator(obs_states)
            discriminator_loss = self.cross_ent_loss(logits, zs.squeeze(-1))
            # discriminator_loss = self.cross_ent_loss(logits_next, zs.squeeze(-1))

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.discriminator_opt.zero_grad()
            discriminator_loss.backward()
            self.discriminator_opt.step()

            self.soft_update_target_network(self.value_network, self.value_target_network)

            mean_reward = rewards.mean().item()
            return q1_loss.item(), policy_loss.item(), -discriminator_loss.item(), mean_reward

    def soft_update_target_network(self, local_network, target_network):
        for target_param, local_param in zip(target_network.parameters(), local_network.parameters()):
            target_param.data.copy_(self.config["tau"] * local_param.data +
                                    (1 - self.config["tau"]) * target_param.data)

    def hard_update_target_network(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())
        self.value_target_network.eval()

    def get_rng_states(self):
        return torch.get_rng_state(), self.memory.get_rng_state()

    def set_rng_states(self, torch_rng_state, random_rng_state):
        torch.set_rng_state(torch_rng_state.to("cpu"))
        self.memory.set_rng_state(random_rng_state)

    def set_policy_net_to_eval_mode(self):
        self.policy_network.eval()

    def set_policy_net_to_cpu_mode(self):
        self.device = torch.device("cpu")
        self.policy_network.to(self.device)
