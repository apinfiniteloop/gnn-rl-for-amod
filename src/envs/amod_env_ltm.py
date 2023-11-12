import os
import subprocess
import json
import random
from copy import deepcopy
from collections import defaultdict
import networkx as nx
import numpy as np


class Scenario:
    """
    Class for AMoD environment scenario. Loads network from json file or generates sample network.
    """
    def __init__(self, use_sample_network=True,
    seed=None, total_time=60, json_file=None, json_hr=9, json_tstep=2, json_regions=None):
        """
        `Scenario` class for AMoD environment. Does all the network loading/generating from sample network/json file.

        Parameters:

        `use_sample_network`: bool, whether to use the sample network or not

        `seed`: int, seed for random number generator

        `total_time`: int, total time of simulation

        `json_file`: str, path to json file

        `json_hr`: int, hour of the day in json file

        `json_tstep`: int, time step in json file
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(self.seed)
        self.total_time = total_time
        if json_file == None or use_sample_network:
            self.is_json=False
            self._load_sample_network()
    

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

        G.add_edge("Or", "A", length=0, q_max=np.inf, k_j=np.inf, w=1e-6, type='origin')
        G.add_edge("A", "B", length=5, q_max=50, k_j=300, w=0.1, type='normal')
        G.add_edge("B", "C", length=10, q_max=50, k_j=300, w=0.1, type='normal')
        G.add_edge("C", "D", length=10, q_max=50, k_j=60, w=0.1, type='normal')
        G.add_edge("C", "D", length=10, q_max=50, k_j=30, w=0.1, type='normal')
        G.add_edge("D", "De", length=0, q_max=np.inf, k_j=np.inf, w=1e-6, type='destination')

        return G

    def _generate_random_demand(self, network, origin, destination, total_time, time_step, demand_scale:tuple=(0, 5)):
        if self.is_json == False:
            origin = "A"
            destination = "D"
        all_paths = list(nx.all_simple_edge_paths(network, source=origin, target=destination))

        path_demands = {path_id: [random.randint(demand_scale[0], demand_scale[1]) for _ in range(total_time // time_step)] for path_id, _ in enumerate(all_paths)}
        # Obtain link demands from path demands
        link_demands = {edge: {time: 0 for time in range(total_time // time_step)} for edge in network.edges(keys=True)}
        for path_id, demand_values in path_demands.items():
            for time, demand in enumerate(demand_values):
                for edge_start, edge_end, edge_key in all_paths[path_id]:
                    link_demands[(edge_start, edge_end, edge_key)][time] += demand
                    # If edge_start is the first node of the path, then add demand to the upstream link of the node with attribute "type" = "origin"
                    if edge_start == all_paths[path_id][0][0]:
                        for edge in network.in_edges(edge_start, keys=True):
                            if network.edges[edge]['type'] == 'origin':
                                link_demands[edge][time] += demand
                    # Same for edge_end
                    if edge_end == all_paths[path_id][-1][1]:
                        for edge in network.out_edges(edge_end, keys=True):
                            if network.edges[edge]['type'] == 'destination':
                                link_demands[edge][time] += demand

        return (path_demands, link_demands), all_paths