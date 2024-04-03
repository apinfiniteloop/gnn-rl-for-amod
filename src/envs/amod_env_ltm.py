import os
import subprocess
import json
import random
from copy import deepcopy
from itertools import product
from collections import defaultdict
from src.misc.utils import EdgeKeyDict, mat2str
import networkx as nx
import numpy as np


class AMoDEnv:
    """
    Class for AMoD environment. Contains all the functions for the environment.
    """

    def __init__(self, scenario, beta=0.2) -> None:
        self.scenario = deepcopy(scenario)
        self.network = scenario.G
        self.gen_cost = defaultdict(
            EdgeKeyDict
        )  # Generalized cost, for every edge and time. gen_cost[i][t] <-> generalized cost of edge i at time t

        # Time related variables
        self.initial_travel_time = scenario.initial_travel_time
        self.time = 0
        self.total_time = scenario.total_time
        self.time_step = scenario.time_step

        # Demand related variables
        self.paths = scenario.all_paths
        self.origins = scenario.origins
        self.destinations = scenario.destinations
        self.pax_demand = (
            scenario.pax_demand
        )  # {(origin, destination): (demand, price)}
        self.path_demands = scenario.path_demands
        self.link_demands = scenario.link_demands
        self.served_demand = defaultdict(dict)
        for o, d in self.pax_demand.keys():
            self.served_demand[o, d] = defaultdict(float)

        # Vehicle count related variables
        self.acc = defaultdict(
            dict
        )  # number of vehicles within each node, acc[t][i] <-> number of vehicles at node i at time t
        self.dacc = defaultdict(
            dict
        )  # number of vehicles arriving at each node, dacc[t][i] <-> number of vehicles arriving at node i at time t
        self.sch = {
            t: {
                i: {(o, d): None for o, d in product(self.network.nodes)}
                for i in self.network.nodes
            }
            for t in range(self.total_time)
        }  # Vehicle schedule. sch[t][i][(o,d)] <-> At time t, the amount of vehicles at node i, about to go from o to d.
        for node in self.network.nodes:
            self.acc[0][node] = self.network.nodes[node]["accInit"]
            self.dacc[0][node] = defaultdict(float)

        # LTM related variables
        self.cvn = defaultdict(EdgeKeyDict)  # Cumulative Vehicle Number
        self.ffs = scenario.ffs
        self.cvn = {  # Initialize N(x, t) for all edges
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
        self.link_mean_travel_time = defaultdict(EdgeKeyDict)
        self.link_mean_travel_time = {
            tuple(edge): {t: 0 for t in range(self.total_time)}
            for edge in self.network.edges
        }
        self.link_traffic_flow = defaultdict(EdgeKeyDict)
        self.link_traffic_flow = {
            tuple(edge): {t: 0 for t in range(self.total_time)}
            for edge in self.network.edges
        }
        # Misc variables
        self.beta = beta * scenario.time_step
        self.info = dict.fromkeys(
            ["revenue", "served_demand", "rebalancing_cost", "operating_cost"], 0
        )
        self.reward = 0

    def calculate_sending_flow(self, edge, edge_attrib, t):
        delta_t = self.time_step
        ffs = self.ffs
        # Source nodes and destination nodes don't have 'w' and 'k_j' attribute.
        if t + delta_t - edge_attrib["length"] / ffs < 0:
            return min(
                self.cvn[tuple(edge)][0][0] - self.cvn[tuple(edge)][t][1],
                edge_attrib["q_max"],
            )
        return min(
            self.cvn[tuple(edge)][t + delta_t - edge_attrib["length"] / ffs][0]
            - self.cvn[tuple(edge)][t][1],
            edge_attrib["q_max"] * delta_t,
        )

    def calculate_receiving_flow(self, edge, edge_attrib, t):
        delta_t = self.time_step
        if edge_attrib["type"] == "destination":
            return np.inf
        try:
            return min(
                self.cvn[tuple(edge)][
                    t + delta_t + edge_attrib["length"] / edge_attrib["w"]
                ][1]
                + edge_attrib["k_j"] * edge_attrib["length"]
                - self.cvn[tuple(edge)][t][0],
                edge_attrib["q_max"] * delta_t,
            )
        except KeyError:
            return min(
                self.cvn[tuple(edge)][self.total_time - 1][1]
                + edge_attrib["k_j"] * edge_attrib["length"]
                - self.cvn[tuple(edge)][t][0],
                edge_attrib["q_max"] * delta_t,
            )

    def generate_path_ids(
        self, network, pax_demand, time, gen_cost, acc, origins, destinations
    ):
        """
        Generate unique path IDs and their costs for IOD paths.

        Args:
        - network: A NetworkX graph representing the network.
        - time_period: The current time period for cost calculation.
        - gen_cost: A dictionary holding the generalized cost of edges.
        - acc: A dictionary holding vehicle accumulations.
        - origins, destinations: Lists of origin and destination node IDs.
        - intermediates: List of intermediate node IDs that each path must pass through.

        Returns:
        - iod_path_tuple: A list of tuples for IOD paths with path IDs and costs.
        - iod_path_dict: A dictionary of dictionaries for IOD paths with path IDs and costs.
        """

        path_id = 0
        iod_path_tuple = []
        iod_path_dict = {}  # iod_path_dict[(i,o,d)][path_id] = (path, cost)

        # Generate IOD paths with unique IDs
        for i, o, d in product(self.acc.keys(), origins, destinations):
            io_paths = nx.all_simple_paths(network, source=i, target=o)
            od_paths = nx.all_simple_paths(network, source=o, target=d)

            for io_path in io_paths:
                for od_path in od_paths:
                    # Combine IO and OD paths, excluding the duplicate occurrence of 'o'
                    combined_path = io_path + od_path[1:]
                    # Calculate the total cost of the combined path
                    total_cost = sum(
                        gen_cost[edge][time]
                        for edge in zip(combined_path[:-1], combined_path[1:])
                    )
                    # Update tuples and dictionaries with the new path and its cost
                    iod_path_tuple.append(
                        (
                            i,  # Vehicle current location
                            o,  # Trip origin
                            d,  # Trip destination
                            path_id,  # Unique Path ID
                            total_cost,  # Total cost of the path (generalized)
                            self.pax_demand[time][(o, d)][0],  # Demand
                            self.pax_demand[time][(o, d)][1],
                        )  # Price
                        if time in self.pax_demand and (o, d) in self.pax_demand[time]
                        else (i, o, d, path_id, total_cost, 0, 0)
                    )
                    if (i, o, d) not in iod_path_dict:
                        iod_path_dict[(i, o, d)] = {}
                    iod_path_dict[path_id] = (combined_path, total_cost, (i, o, d))
                    path_id += 1

        return iod_path_tuple, iod_path_dict

    def format_for_opl(self, value):
        """Format Python data types into strings that OPL can parse."""
        if isinstance(value, list):
            return "[" + " ".join(self.format_for_opl(v) for v in value) + "]"
        elif isinstance(value, tuple):
            return "<" + ",".join(self.format_for_opl(v) for v in value) + ">"
        elif isinstance(value, dict):
            return (
                "{"
                + " ".join(f"{k}={self.format_for_opl(v)}" for k, v in value.items())
                + "}"
            )
        else:
            return str(value)

    def matching(self, CPLEXPATH=None, PATH="", platform="win"):
        """

        Matching function for AMoD environment. Uses CPLEX to solve the optimization problem.

        Parameters:

        `CPLEXPATH`: str, path to CPLEX executable

        `PATH`: str, path to store CPLEX output
        """
        t = self.time
        demand_attr = [
            (i, j, self.pax_demand[t][(i, j)][0], self.pax_demand[t][(i, j)][1])
            for i, j in product(self.origins, self.destinations)
            if t in self.pax_demand.keys()
            and (i, j) in self.pax_demand[t].keys()
            and self.pax_demand[t][(i, j)][0] > 1e-6
        ]  # Demand attributes, (origin, destination, demand, price)
        acc_tuple = [
            (i, self.acc[i][t + 1]) for i in self.acc
        ]  # Accumulation attributes
        iod_path_tuple, iod_path_dict = self.generate_path_ids(
            self.network, t, self.gen_cost, self.acc, self.origins, self.destinations
        )

        mod_path = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
        matching_path = (
            os.getcwd().replace("\\", "/")
            + "/saved_files/cplex_logs/matching/"
            + PATH
            + "/"
        )
        if not os.path.exists(matching_path):
            os.makedirs(matching_path)
        data_file = matching_path + "data_{}.dat".format(t)
        res_file = matching_path + "res_{}.dat".format(t)
        with open(data_file, "w", encoding="UTF-8") as f:
            # Writing demandAttr
            f.write(f"demandAttr = {self.format_for_opl(demand_attr)};\n")
            # Writing accInitTuple
            f.write(f"accInitTuple = {self.format_for_opl(acc_tuple)};\n")
            # Writing iodPathTuple
            f.write(f"iodPathTuple = {self.format_for_opl(iod_path_tuple)};\n")
        mod_file = mod_path + "matching.mod"
        if CPLEXPATH is None:
            CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
        my_env = os.environ.copy()
        if platform == "mac":
            my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
        else:
            my_env["LD_LIBRARY_PATH"] = CPLEXPATH
        out_file = matching_path + "out_{}.dat".format(t)
        with open(out_file, "w", encoding="UTF-8") as output_f:
            subprocess.check_call(
                [CPLEXPATH + "oplrun", mod_file, data_file], stdout=output_f, env=my_env
            )
        output_f.close()
        flow = defaultdict(float)
        # Retrieve and process the result file. TODO: Write it.
        with open(resfile, "r", encoding="utf8") as file:
            for row in file:
                item = row.replace("e)", ")").strip().strip(";").split("=")
                if item[0] == "flow":
                    values = item[1].strip(")]").strip("[(").split(")(")
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, o, d, pid, f = v.split(",")
                        flow[int(i), int(o), int(d), int(pid)] = float(f)
        pax_action = {
            (i, o, d, pid): flow[i, o, d, pid] if (i, o, d, pid) in flow else 0
            for i, o, d, pid, _, _, _ in iod_path_tuple
        }
        # pax_action: Retreived from OPL and formulated to be used in pax_step and LTM. pax_action[(i,o,d,path_id)] = flow starting from i to o to d, using path_id. For path_id correspondence see iod_path_tuple.
        # iod_path_dict: Formulated to be used in pax_step and LTM. iod_path_dict[path_id] = (path, cost)
        return pax_action, iod_path_dict

    def pax_step(self, paxAction=None, CPLEXPATH=None, PATH="", platform="win", taylor_params=None):
        t = self.time
        delta_t = self.time_step
        # Do a step in passenger matching
        for i in self.network.nodes:
            self.acc[t + 1][i] = self.acc[t][i]
        self.info["served_demand"] = 0  # initialize served demand
        self.info["operating_cost"] = 0  # initialize operating cost
        self.info["revenue"] = 0
        self.info["rebalancing_cost"] = 0
        if (
            paxAction is None
        ):  # default matching algorithm used if isMatching is True, matching method will need the information of self.acc[t+1], therefore this part cannot be put forward
            paxAction, paxPathDict = self.matching(
                CPLEXPATH=CPLEXPATH, PATH=PATH, platform=platform
            )
        self.paxAction = paxAction  # pax_action[(i, j)][path_id] = flow
        # Passenger serving
        # Obtain path demands from i to o to d, traversing path_id
        # iod_path_demands = {pid: [flow for _ in range(self.total_time // self.time_step)] if flow > 1e-6 else [0 for _ in range(self.total_time // self.time_step)] for (_, _, _, pid), flow in paxAction.items()}
        iod_path_demands = {
            pid: flow if flow > 1e-6 else 0
            for (_, _, _, pid), flow in paxAction.items()
        }
        self.path_demands[paxPathDict[pid][0]][t] = iod_path_demands[pid]
        # Obtain link demands from path demands
        if self.link_demands is None:
            self.link_demands = {
                edge: {time: 0 for time in range(self.total_time // self.time_step)}
                for edge in self.network.edges(keys=True)
            }
        for pid, flow in iod_path_demands.items():
            self.path_demands[paxPathDict[pid][0]][t+delta_t] += flow
            for edge_start, edge_end, edge_key in paxPathDict[pid][0]:
                self.link_demands[(edge_start, edge_end, edge_key)][t] += flow
                # If edge_start is the first node of the path, then add demand to the upstream link of the node with attribute "type" = "origin"
                if edge_start == paxPathDict[pid][0][0][0]:
                    for edge in self.network.in_edges(edge_start, keys=True):
                        if self.network.edges[edge]["type"] == "origin":
                            self.link_demands[edge][t] += flow
                # Same for edge_end
                if edge_end == paxPathDict[pid][0][-1][1]:
                    for edge in self.network.out_edges(edge_end, keys=True):
                        if self.network.edges[edge]["type"] == "destination":
                            self.link_demands[edge][t] += flow
            # Update the served demand and the revenue
            o, d = paxPathDict[pid][2][1], paxPathDict[pid][2][2]
            assert iod_path_demands[pid] < self.acc[o][t + 1] + 1e-3
            self.servedDemand[o, d][t] = self.iod_path_demands[pid]
            self.paxFlow[o, d][t + self.demandTime[pid][t]] = self.iod_path_demands[pid]
            self.info["operating_cost"] += (
                self.demandTime[pid][t] * self.beta * self.iod_path_demands[pid]
            )  # TODO: Need definition for self.demandTime
            self.acc[i][t + 1] -= self.iod_path_demands[pid]
            self.info["served_demand"] += self.servedDemand[pid][t]
            self.dacc[d][t + self.demandTime[pid][t]] += self.paxFlow[pid][
                t + self.demandTime[pid][t]
            ]
            self.reward += self.iod_path_demand[pid] * (
                self.pax_demand[t][(o, d)][1] - self.beta * self.iod_path_demand[pid]
            )
            self.info["revenue"] += (
                self.iod_path_demand[pid] * self.pax_demand[t][(o, d)][1]
            )

        self.obs = (self.acc, self.time, self.dacc, self.pax_demand)
        done = False
        return self.obs, max(0, self.reward), done, self.info

    def reb_step(self, rebAction):
        t = self.time
        self.reward = 0
        self.rebAction = rebAction
        # Rebalancing
        for k, (edge_start, edge_end, edge_key) in enumerate(
            self.network.edges(keys=True, data=False)
        ):
            self.rebAction[k] = min(self.acc[i][t + 1], rebAction[k])
            self.rebFlow[edge_start, edge_end, edge_key][
                t + self.rebTime[edge_start, edge_end, edge_key][t]
            ] = self.rebAction[k]
            self.acc[edge_start][t + 1] -= self.rebAction[k]
            self.dacc[edge_end][
                t + self.rebTime[edge_start, edge_end, edge_key][t]
            ] += self.rebFlow[edge_start, edge_end, edge_key][
                t + self.rebTime[edge_start, edge_end, edge_key][t]
            ]
            self.info["rebalancing_cost"] += (
                self.rebTime[edge_start, edge_end, edge_key][t]
                * self.beta
                * self.rebAction[k]
            )
            self.info["operating_cost"] += (
                self.rebTime[edge_start, edge_end, edge_key][t]
                * self.beta
                * self.rebAction[k]
            )
            self.reward -= (
                self.rebAction[k]
                * self.beta
                * self.rebTime[edge_start, edge_end, edge_key][t]
            )
        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing
        for k, (edge_start, edge_end, edge_key) in enumerate(
            self.network.edges(keys=True, data=False)
        ):
            if (edge_start, edge_end, edge_key) in self.rebFlow and t in self.rebFlow[
                edge_start, edge_end, edge_key
            ]:
                self.acc[edge_end][t + 1] += self.rebFlow[
                    edge_start, edge_end, edge_key
                ][t]
            if (edge_start, edge_end, edge_key) in self.paxFlow and t in self.paxFlow[
                edge_start, edge_end, edge_key
            ]:
                self.acc[edge_end][t + 1] += self.paxFlow[
                    edge_start, edge_end, edge_key
                ][
                    t
                ]  # this means that after pax arrived, vehicles can only be rebalanced in the next time step, let me know if you have different opinion
        
        self.time += 1
        self.obs = (self.acc, self.time, self.dacc, self.pax_demand)
        done = self.time == self.total_time
        return self.obs, self.reward, done, self.info

    def ltm_step(self, use_ctm_at_merge=False):
        t = self.time
        delta_t = self.time_step
        # Do a step in LTM
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
                out_node = tuple(out_edges)[0][1]
                transition_flow = min(
                    sum(
                        # [
                        #     self.path_demands[i][t + delta_t]
                        #     for i in self.path_demands.keys()
                        # ]
                        self.path_demands[path][t+delta_t] for path in self.path_demands.keys() if path[0][0] == out_node else 0
                    ),
                    self.receiving_flow[tuple(out_edges)[0]][t],
                )
                self.cvn[tuple(out_edges)[0]][t + delta_t][0] = (
                    self.cvn[tuple(out_edges)[0]][t][0] + transition_flow
                )
            # If is homogenous node, update N(x, t) for outgoing edges
            elif len(in_edges) == 1 and len(out_edges) == 1:
                transition_flow = min(
                    self.sending_flow[tuple(in_edges)[0]][t],
                    self.receiving_flow[tuple(out_edges)[0]][t],
                )
                self.cvn[tuple(in_edges)[0]][t + delta_t][1] = (
                    self.cvn[tuple(in_edges)[0]][t][1] + transition_flow
                )
                self.cvn[tuple(out_edges)[0]][t + delta_t][0] = (
                    self.cvn[tuple(out_edges)[0]][t][0] + transition_flow
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
                    self.cvn[tuple(edge)][t + delta_t][0] = self.cvn[tuple(edge)][t][
                        0
                    ] + round(transition_flow)
                    sum_transition_flow += transition_flow
                self.cvn[tuple(in_edges)[0]][t + delta_t][1] = self.cvn[
                    tuple(in_edges)[0]
                ][t][1] + round(sum_transition_flow)
            # If is merge node, update N(x, t) for outgoing edges
            elif len(in_edges) > 1 and len(out_edges) == 1:
                sum_transition_flow = 0
                for edge in in_edges:
                    if use_ctm_at_merge:
                        transition_flow = sorted(
                            [
                                self.sending_flow[edge][t],
                                self.receiving_flow[tuple(out_edges)[0]][t]
                                - (
                                    sum(self.sending_flow[k][t] for k in in_edges)
                                    - self.sending_flow[edge][t]
                                ),
                                p * self.receiving_flow[tuple(out_edges)[0]][t],
                            ]
                        )[1]
                    else:
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
                        self.cvn[edge][t + delta_t][1] = (
                            self.cvn[edge][t][1] + transition_flow
                        )
                        sum_transition_flow += transition_flow
                self.cvn[tuple(out_edges)[0]][t + delta_t][0] = (
                    self.cvn[tuple(out_edges)[0]][t][0] + sum_transition_flow
                )
            # If is destination node, update N(x, t) for outgoing edges
            elif len(in_edges) == 1 and len(out_edges) == 0:
                transition_flow = self.sending_flow[tuple(in_edges)[0]][t]
                self.cvn[tuple(in_edges)[0]][t + delta_t][1] = (
                    self.cvn[tuple(in_edges)[0]][t][1] + transition_flow
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
                        self.cvn[out_edge][t + delta_t][0] = (
                            self.cvn[out_edge][t][0] + transition_flow
                        )
                        self.cvn[in_edge][t + delta_t][1] = (
                            self.cvn[in_edge][t][1] + transition_flow
                        )
        self.time += self.time_step


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
        self.origins = {}  # {time: [origins]}
        self.destinations = {}  # {time: [destinations]}

    def _load_sample_network(self, name):
        """
        Loads sample network.
        """
        if name == 'sample':
            G = nx.MultiDiGraph()

            # G.add_node("Or")
            G.add_node("A")
            G.add_node("B")
            G.add_node("C")
            G.add_node("D")
            # G.add_node("De")

            # G.add_edge("Or", "A", length=0, q_max=np.inf, k_j=np.inf, w=1e-6, type="origin")
            G.add_edge("A", "B", length=5, q_max=50, k_j=300, w=0.1, type="normal")
            G.add_edge("B", "C", length=10, q_max=50, k_j=300, w=0.1, type="normal")
            G.add_edge("C", "D", length=10, q_max=50, k_j=60, w=0.1, type="normal")
            G.add_edge("C", "D", length=10, q_max=50, k_j=30, w=0.1, type="normal")
            # G.add_edge(
            #     "D", "De", length=0, q_max=np.inf, k_j=np.inf, w=1e-6, type="destination"
            # )
        elif name == 'sioux_falls':
            raise NotImplementedError

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
            network.add_node(o + "*")
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
            network.add_node(d + "*")
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

    def _generate_original_acc(self, network, acc_scale: tuple = (0, 5)):
        for node in network.nodes:
            node["accInit"] = random.randint(acc_scale[0], acc_scale[1])

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
