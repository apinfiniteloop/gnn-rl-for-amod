import os
import subprocess
import json
import random
from copy import deepcopy
from itertools import product
from collections import defaultdict
from src.misc.utils import EdgeKeyDict
import networkx as nx
import numpy as np


class AMoDEnv:
    def __init__(self, scenario, beta=0.2) -> None:
        self.scenario = deepcopy(scenario)
        self.network = scenario.G
        self.origins = scenario.origins
        self.destinations = scenario.destinations
        self.pax_demand = scenario.pax_demand
        self.path_demands = scenario.path_demands
        self.link_demands = scenario.link_demands
        self.acc = defaultdict(
            dict
        )  # number of vehicles within each region, key: i - region, t - time
        self.dacc = defaultdict(
            dict
        )  # number of vehicles arriving at each region, key: i - region, t - time
        self.paths = scenario.all_paths
        self.beta = beta
        self.initial_travel_time = scenario.initial_travel_time
        self.time = 0
        self.total_time = scenario.total_time
        self.time_step = scenario.time_step
        self.N = defaultdict(EdgeKeyDict)  # Cumulative Vehicle Number
        self.ffs = scenario.ffs
        self.N = {  # Initialize N(x, t) for all edges
            tuple(edge): {
                t: {0: 0, 1: 0}  # Upstream end  # Downstream end
                for t in range(self.total_time)
            }
            for edge in self.network.edges
        }  # N(x, t)<->N[x][t][0/1]
        self.sending_flow = defaultdict(EdgeKeyDict)  # Sending Flow
        self.sending_flow = {  # Initialize sending flow for all edges
            tuple(edge): {t: 0 for t in range(self.total_time)}
            for edge in self.network.edges
        }  # S_i(t) <-> sending_flow[i][t]
        self.receiving_flow = defaultdict(EdgeKeyDict)  # Receiving Flow
        self.receiving_flow = {  # Initialize receiving flow for all edges
            tuple(edge): {t: 0 for t in range(self.total_time)}
            for edge in self.network.edges
        }  # R_i(t) <-> receiving_flow[i][t]
        self.node_transition_demand = defaultdict(
            EdgeKeyDict
        )  # Node Transition Demand, for storing demand of edge i to edge j, forall i in in_edges and j in out_edges
        self.node_transition_demand = (
            {  # Initialize node transition demand for all nodes
                node: {
                    i: {
                        j: {t: 0 for t in range(self.total_time)}
                        for j in self.network.out_edges(node, keys=True)
                    }
                    for i in self.network.in_edges(node, keys=True)
                }
                for node in self.network.nodes
            }
        )
        # Calculate node transition demand based on path demand
        for path_id, path in enumerate(self.paths):
            for time in range(self.total_time):
                for i in range(len(path) - 1):
                    self.node_transition_demand[path[i][1]][path[i]][path[i + 1]][
                        time
                    ] += self.path_demands[path_id][time]
        for edge in self.network.edges(keys=True):
            if self.network.edges[edge]["type"] == "origin":
                self.N[edge][0][0] = self.link_demands[edge][0]

    def calculate_sending_flow(self, edge, edge_attrib, t):
        delta_t = self.time_step
        ffs = self.ffs
        # Source nodes and destination nodes don't have 'w' and 'k_j' attribute.
        if t + delta_t - edge_attrib["length"] / ffs < 0:
            return min(
                self.N[tuple(edge)][0][0] - self.N[tuple(edge)][t][1],
                edge_attrib["q_max"],
            )
        return min(
            self.N[tuple(edge)][t + delta_t - edge_attrib["length"] / ffs][0]
            - self.N[tuple(edge)][t][1],
            edge_attrib["q_max"] * delta_t,
        )

    def calculate_receiving_flow(self, edge, edge_attrib, t):
        delta_t = self.time_step
        if edge_attrib["type"] == "destination":
            return np.inf
        try:
            return min(
                self.N[tuple(edge)][
                    t + delta_t + edge_attrib["length"] / edge_attrib["w"]
                ][1]
                + edge_attrib["k_j"] * edge_attrib["length"]
                - self.N[tuple(edge)][t][0],
                edge_attrib["q_max"] * delta_t,
            )
        except KeyError:
            return min(
                self.N[tuple(edge)][self.total_time - 1][1]
                + edge_attrib["k_j"] * edge_attrib["length"]
                - self.N[tuple(edge)][t][0],
                edge_attrib["q_max"] * delta_t,
            )

    def matching(self, CPLEXPATH=None, PATH=""):
        """

        Matching function for AMoD environment. Uses CPLEX to solve the optimization problem.

        Parameters:

        `CPLEXPATH`: str, path to CPLEX executable

        `PATH`: str, path to store CPLEX output
        """
        t = self.time
        demandAttr = [
            (i, j, self.pax_demand[t][(i, j)][0], self.pax_demand[t][(i, j)][1])
            for i, j in product(self.origins, self.destinations)
            if t in self.pax_demand.keys() and (i, j) in self.pax_demand[t].keys()
        ]  # Demand attributes, (origin, destination, demand, price)

    def ltm_step(self):
        t = self.time
        delta_t = self.time_step
        for node in self.network.nodes:
            # For each node, calculate the sending flow and receiving flow of its connected edges
            for edge in set(
                out_edges := self.network.out_edges(nbunch=node, keys=True)
            ) | set(in_edges := self.network.in_edges(nbunch=node, keys=True)):
                self.sending_flow[tuple(edge)][t] = self.calculate_sending_flow(
                    edge, self.network.edges[edge], t
                )
                self.receiving_flow[tuple(edge)][t] = self.calculate_receiving_flow(
                    edge, self.network.edges[edge], t
                )
            # If is origin node, update N(x, t) for outgoing edges
            if len(in_edges) == 0 and len(out_edges) == 1:
                transition_flow = min(
                    sum(
                        [
                            self.path_demands[i][t + delta_t]
                            for i in self.path_demands.keys()
                        ]
                    ),
                    self.receiving_flow[tuple(out_edges)[0]][t],
                )
                self.N[tuple(out_edges)[0]][t + delta_t][0] = (
                    self.N[tuple(out_edges)[0]][t][0] + transition_flow
                )
            # If is homogenous node, update N(x, t) for outgoing edges
            elif len(in_edges) == 1 and len(out_edges) == 1:
                transition_flow = min(
                    self.sending_flow[tuple(in_edges)[0]][t],
                    self.receiving_flow[tuple(out_edges)[0]][t],
                )
                self.N[tuple(in_edges)[0]][t + delta_t][1] = (
                    self.N[tuple(in_edges)[0]][t][1] + transition_flow
                )
                self.N[tuple(out_edges)[0]][t + delta_t][0] = (
                    self.N[tuple(out_edges)[0]][t][0] + transition_flow
                )
            # If is split node, update N(x, t) for outgoing edges
            elif len(in_edges) == 1 and len(out_edges) > 1:
                sum_transition_flow = 0
                for edge in out_edges:
                    try:
                        p = (
                            self.link_demands[edge][t]
                            / self.link_demands[tuple(in_edges)[0]][t]
                        )
                        transition_flow = p * min(
                            self.sending_flow[tuple(in_edges)[0]][t],
                            min(
                                [
                                    self.receiving_flow[e][t]
                                    / (
                                        self.link_demands[e][t]
                                        / self.link_demands[tuple(in_edges)[0]][t]
                                    )
                                    for e in out_edges
                                ]
                            ),
                        )
                    except ZeroDivisionError:
                        transition_flow = 0
                    self.N[tuple(edge)][t + delta_t][0] = self.N[tuple(edge)][t][
                        0
                    ] + round(transition_flow)
                    sum_transition_flow += transition_flow
                self.N[tuple(in_edges)[0]][t + delta_t][1] = self.N[tuple(in_edges)[0]][
                    t
                ][1] + round(sum_transition_flow)
            # If is merge node, update N(x, t) for outgoing edges
            elif len(in_edges) > 1 and len(out_edges) == 1:
                sum_transition_flow = 0
                for edge in in_edges:
                    # if self.link_demands[tuple(out_edges)[0]][t] == 0:
                    #     continue
                    # self.disaggregate_demand(edge, t)
                    # Daganzo CTM model, lacks disaggregation of sending flow.
                    # transition_flow = sorted([self.sending_flow[edge][t], self.receiving_flow[tuple(out_edges)[0]][t]-(sum(self.sending_flow[k][t] for k in in_edges)-self.sending_flow[edge][t]), p*self.receiving_flow[tuple(out_edges)[0]][t]])[1]
                    # Jin and Zhang fairness model.
                    try:
                        p = (
                            self.link_demands[edge][t]
                            / self.link_demands[tuple(out_edges)[0]][t]
                        )
                        transition_flow = min(
                            self.sending_flow[edge][t],
                            self.receiving_flow[tuple(out_edges)[0]][t]
                            * self.sending_flow[edge][t]
                            / (sum([self.sending_flow[e][t] for e in in_edges])),
                        )
                    except ZeroDivisionError:
                        transition_flow = 0
                    self.N[edge][t + delta_t][1] = self.N[edge][t][1] + transition_flow
                    sum_transition_flow += transition_flow
                self.N[tuple(out_edges)[0]][t + delta_t][0] = (
                    self.N[tuple(out_edges)[0]][t][0] + sum_transition_flow
                )
            # If is destination node, update N(x, t) for outgoing edges
            elif len(in_edges) == 1 and len(out_edges) == 0:
                transition_flow = self.sending_flow[tuple(in_edges)[0]][t]
                self.N[tuple(in_edges)[0]][t + delta_t][1] = (
                    self.N[tuple(in_edges)[0]][t][1] + transition_flow
                )
            # If is normal node, update N(x, t) for outgoing edges
            elif len(in_edges) > 1 and len(out_edges) > 1:
                sum_inout = sum(
                    [
                        self.node_transition_demand[node][i][j][t]
                        for i in in_edges
                        for j in out_edges
                    ]
                )
                for in_edge in in_edges:
                    for out_edge in out_edges:
                        p = (
                            self.node_transition_demand[node][in_edge][out_edge][t]
                            / sum_inout
                        )
                        transition_flow = p * min(
                            min(
                                [
                                    self.receiving_flow[out_edge][t]
                                    * self.sending_flow[in_edge][t]
                                    / (
                                        sum(
                                            [
                                                self.node_transition_demand[node][i][
                                                    out_edge
                                                ]
                                                * self.sending_flow[i][t]
                                                for i in in_edges
                                            ]
                                        )
                                    )
                                ]
                            ),
                            self.sending_flow[in_edge][t],
                        )
                        self.N[out_edge][t + delta_t][0] = (
                            self.N[out_edge][t][0] + transition_flow
                        )
                        self.N[in_edge][t + delta_t][1] = (
                            self.N[in_edge][t][1] + transition_flow
                        )
        self.time += delta_t


class Scenario:
    """
    Class for AMoD environment scenario. Loads network from json file or generates sample network.
    """

    def __init__(
        self,
        use_sample_network=True,
        seed=None,
        total_time=60,
        time_step=1,
        json_file=None,
        json_hr=9,
        json_tstep=2,
        json_regions=None,
        ffs=0.2,
    ):
        """
        `Scenario` class for AMoD environment. Does all the network loading/generating from sample network/json file.

        Parameters:

        `use_sample_network`: bool, whether to use the sample network or not

        `seed`: int, seed for random number generator

        `total_time`: int, total time of simulation

        `json_file`: str, path to json file

        `json_hr`: int, hour of the day in json file

        `json_tstep`: int, time step in json file

        `ffs`: float, free flow speed of network
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        self.total_time = total_time
        self.time_step = time_step
        if json_file is None or use_sample_network:
            self.is_json = False
            self.G = self._load_sample_network()
            # (
            #     self.path_demands,
            #     self.link_demands,
            # ), self.all_paths = self._generate_random_demand(self.G, self.total_time, 2)
            self.pax_demand = self._generate_random_demand(
                self.G, self.total_time, self.time_step
            )
        else:
            # If Using json file. TODO: Need a json file to complete.
            raise NotImplementedError
        self.initial_travel_time = {
            edge: self.G.edges[edge]["length"] / ffs for edge in self.G.edges(keys=True)
        }
        self.ffs = ffs
        self.origins = []
        self.destinations = []

    def _load_sample_network(self):
        """
        Loads sample network.
        """
        G = nx.MultiDiGraph()

        G.add_node("Or")
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")
        G.add_node("D")
        G.add_node("De")

        # G.add_edge("Or", "A", length=0, q_max=np.inf, k_j=np.inf, w=1e-6, type="origin")
        G.add_edge("A", "B", length=5, q_max=50, k_j=300, w=0.1, type="normal")
        G.add_edge("B", "C", length=10, q_max=50, k_j=300, w=0.1, type="normal")
        G.add_edge("C", "D", length=10, q_max=50, k_j=60, w=0.1, type="normal")
        G.add_edge("C", "D", length=10, q_max=50, k_j=30, w=0.1, type="normal")
        # G.add_edge(
        #     "D", "De", length=0, q_max=np.inf, k_j=np.inf, w=1e-6, type="destination"
        # )

        return G

    def _generate_dummy_od_links(
        self,
        network,
        origins,
        destinations,
        dummy_length=0,
        dummy_q_max=np.inf,
        dummy_k_j=np.inf,
        dummy_w=1e-6,
    ):
        for o in origins:
            network.add_edge(
                o + "*",
                o,
                length=dummy_length,
                q_max=dummy_q_max,
                k_j=dummy_k_j,
                w=dummy_w,
                type="origin",
            )
        for d in destinations:
            network.add_edge(
                d,
                d + "*",
                length=dummy_length,
                q_max=dummy_q_max,
                k_j=dummy_k_j,
                w=dummy_w,
                type="destination",
            )

    def _generate_random_demand(
        self,
        network,
        total_time,
        time_step,
        demand_scale: tuple = (0, 5),
        price_scale: tuple = (10, 30),
    ):
        if not self.is_json:
            self.origins.append("A")
            self.destinations.append("D")
            self._generate_dummy_od_links(network, self.origins, self.destinations)

        self.pax_demand = {
            time_step: {
                (
                    origin,
                    destination,
                ): (
                    random.randint(demand_scale[0], demand_scale[1]),
                    random.randint(price_scale[0], price_scale[1]),
                )
                for origin, destination in product(self.origins, self.destinations)
            }
            for time_step in range(self.total_time)
        }

        # if not self.is_json:
        #     self.origins.append("A")
        #     self.destinations.append("D")
        # all_paths = list(
        #     [
        #         nx.all_simple_edge_paths(network, source=o, target=self.destinations)
        #         for o in self.origins
        #     ]
        # )

        # path_demands = {
        #     path_id: [
        #         random.randint(demand_scale[0], demand_scale[1])
        #         for _ in range(total_time // time_step)
        #     ]
        #     for path_id, _ in enumerate(all_paths)
        # }
        # # Obtain link demands from path demands
        # link_demands = {
        #     edge: {time: 0 for time in range(total_time // time_step)}
        #     for edge in network.edges(keys=True)
        # }
        # for path_id, demand_values in path_demands.items():
        #     for time, demand in enumerate(demand_values):
        #         for edge_start, edge_end, edge_key in all_paths[path_id]:
        #             link_demands[(edge_start, edge_end, edge_key)][time] += demand
        #             # If edge_start is the first node of the path, then add demand to the upstream link of the node with attribute "type" = "origin"
        #             if edge_start == all_paths[path_id][0][0]:
        #                 for edge in network.in_edges(edge_start, keys=True):
        #                     if network.edges[edge]["type"] == "origin":
        #                         link_demands[edge][time] += demand
        #             # Same for edge_end
        #             if edge_end == all_paths[path_id][-1][1]:
        #                 for edge in network.out_edges(edge_end, keys=True):
        #                     if network.edges[edge]["type"] == "destination":
        #                         link_demands[edge][time] += demand

        # return (path_demands, link_demands), all_paths
