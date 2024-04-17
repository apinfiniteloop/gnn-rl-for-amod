"""
A2C-GNN
-------
This file contains the A2C-GNN specifications. In particular, we implement:
(1) GNNParser
    Converts raw environment observations to agent inputs (s_t).
(2) GNNActor:
    Policy parametrized by Graph Convolution Networks (Section III-C in the paper)
(3) GNNCritic:
    Critic parametrized by Graph Convolution Networks (Section III-C in the paper)
(4) A2C:
    Advantage Actor Critic algorithm using a GNN parametrization for both Actor and Critic.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import networkx as nx
from torch.distributions import Dirichlet
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import grid
from collections import namedtuple

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])
args = namedtuple("args", ("render", "gamma", "log_interval"))
args.render = True
args.gamma = 0.97
args.log_interval = 10

#########################################
############## PARSER ###################
#########################################


class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(
        self,
        env,
        use_grid=False,
        T=10,
        grid_h=4,
        grid_w=4,
        scale_factor=0.01,
        network=None,
    ):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.use_grid = use_grid
        if self.use_grid:
            self.grid_h = grid_h
            self.grid_w = grid_w

    def parse_obs(self, obs):
        x = (
            torch.cat(
                (
                    torch.tensor(
                        [
                            obs[4][n][self.env.time + 1] * self.s
                            for n in [
                                edge
                                for edge in self.env.network.edges(keys=True)
                                if edge[0][-1] != "*" and edge[1][-1] != "*"
                            ]
                        ]
                    )
                    .view(1, 1, self.env.nedges)
                    .float(),
                    torch.tensor(
                        [
                            obs[5][n][self.env.time] * self.s
                            for n in [
                                edge
                                for edge in self.env.network.edges(keys=True)
                                if edge[0][-1] != "*" and edge[1][-1] != "*"
                            ]
                        ]
                    )
                    .view(1, 1, self.env.nedges)
                    .float(),
                    torch.tensor(
                        [
                            [
                                (
                                    obs[0][edge[0]][self.env.time + 1]
                                    + self.env.dacc[edge[0]][t]
                                )
                                * self.s
                                for edge in self.env.network.edges(keys=True)
                                if edge[0][-1] != "*" and edge[1][-1] != "*"
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nedges)
                    .float(),
                    torch.tensor(
                        [
                            [
                                sum(
                                    [
                                        (
                                            self.env.scenario.demand_input[i[0], j][t]
                                            if t
                                            in self.env.scenario.demand_input[i[0], j]
                                            else 0
                                        )
                                        * (
                                            self.env.price[i[0], j][t]
                                            if t in self.env.price[i[0], j]
                                            else 0
                                        )
                                        * self.s
                                        for j in self.env.region
                                    ]
                                )
                                for i in self.env.network.edges(keys=True)
                                if i[0][-1] != "*" and i[1][-1] != "*"
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nedges)
                    .float(),
                ),
                dim=1,
            )
            .squeeze(0)
            .view(22, self.env.nedges)
            .T
        )
        if self.use_grid:
            edge_index, pos_coord = grid(height=self.grid_h, width=self.grid_w)
        else:
            if type(self.env.network) == nx.MultiDiGraph:
                network = nx.DiGraph()
                for node in self.env.network.nodes:
                    if node[-1] != "*":
                        network.add_node(node)
                for edge in self.env.network.edges:
                    if (
                        self.env.network.edges[edge]["type"] != "origin"
                        and self.env.network.edges[edge]["type"] != "destination"
                    ):
                        network.add_edge(edge[0], edge[1])
            pyg_graph = from_networkx(network)
            edge_index = pyg_graph.edge_index
        data = Data(x, edge_index)
        return data


#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    `D` is the dimensions of the parameter space for the Dirichlet distribution. Defaults to 1.
    `T` is the dimensions of the parameter space for the Taylor's series. Defaults to 0 (disabled). Change to 1 if considering the approximation of BPR.
    """

    def __init__(self, in_channels, out_channels, D=1, T=3):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, D + T)

    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


#########################################
############## CRITIC ###################
#########################################


class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, 1)

    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = torch.sum(x, dim=0)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


#########################################
############## A2C AGENT ################
#########################################


class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,
        input_size,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
        estimate_bpr=True,
    ):
        super(A2C, self).__init__()
        self.env = env
        self.eps = eps
        self.input_size = input_size
        self.hidden_size = input_size
        self.device = device

        self.D = 1
        if estimate_bpr:
            self.T = 3
        else:
            self.T = 0

        self.actor = GNNActor(self.input_size, self.hidden_size, self.D, self.T)
        self.critic = GNNCritic(self.input_size, self.hidden_size)
        self.obs_parser = GNNParser(self.env)

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def forward(self, obs, jitter=1e-20):
        """
        forward of both actor and critic
        """
        # parse raw environment data in model format
        x = self.parse_obs(obs).to(self.device)

        # actor: computes concentration parameters of a Dirichlet distribution
        a_out = self.actor(x)
        concentration = F.softplus(a_out[:, : self.D]).reshape(-1) + jitter
        # Handle case when T=0 (no Taylor series parameters)
        if (
            a_out.shape[1] > self.D
        ):  # Check if there are more outputs beyond Dirichlet parameters
            taylor_params = F.softplus(a_out[:, self.D :]) + jitter
            # Extract Taylor series parameters
        else:
            taylor_params = None  # Or an appropriate default value indicating no Taylor series parameters

        # critic: estimates V(s_t)
        value = self.critic(x)
        return concentration, taylor_params, value

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, obs):
        concentration, taylor_params, value = self.forward(obs)

        m = Dirichlet(concentration)
        n = Dirichlet(taylor_params)

        action = m.sample()
        taylor_action = n.sample()
        self.saved_actions.append(
            SavedAction(m.log_prob(action) + n.log_prob(taylor_action), value)
        )
        if taylor_params is None:
            return list(action.cpu().numpy())
        else:
            return list(action.cpu().numpy()), list(
                taylor_params.cpu().detach().numpy()
            )

    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + args.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(
                F.smooth_l1_loss(value, torch.tensor([R]).to(self.device))
            )

        # take gradient steps
        self.optimizers["a_optimizer"].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        self.optimizers["a_optimizer"].step()

        self.optimizers["c_optimizer"].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        self.optimizers["c_optimizer"].step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def configure_optimizers(self):
        optimizers = dict()
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=1e-3)
        optimizers["c_optimizer"] = torch.optim.Adam(critic_params, lr=1e-3)
        return optimizers

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model"])
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
