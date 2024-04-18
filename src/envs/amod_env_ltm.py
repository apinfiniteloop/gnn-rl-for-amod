import os
import subprocess
import json
import random
import math

# import time as tm
from copy import deepcopy
from itertools import product
from collections import defaultdict
from src.misc.utils import EdgeKeyDict, DefaultList  # , mat2str
from src.misc.caching import PathCacheManager
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
            dict
        )  # Generalized cost, for every edge and time. gen_cost[i][t] <-> generalized cost of edge i at time t

        # Time related variables
        self.initial_travel_time = scenario.initial_travel_time
        self.time = 0
        self.total_time = scenario.total_time
        self.time_step = scenario.time_step

        # Demand related variables
        # self.paths = scenario.all_paths
        self.region = [node for node in self.network.nodes if node[-1] != "*"]
        self.nregions = scenario.nregions
        self.nedges = scenario.nedges
        self.origins = scenario.origins
        self.destinations = scenario.destinations
        self.od_pairs = scenario.od_pairs
        self.price = scenario.price
        self.extra_timesteps = 10
        self.pax_demand = (
            scenario.pax_demand
        )  # {(origin, destination): (demand, price)}
        # self.path_demands = scenario.path_demands
        # self.link_demands = scenario.link_demands
        self.path_demands = defaultdict(lambda: defaultdict(int))
        # self.link_demands = defaultdict(dict)
        self.link_demands = {
            edge: {time: 0 for time in range(self.total_time // self.time_step + 1)}
            for edge in self.network.edges(keys=True)
        }
        self.served_demand = defaultdict(dict)
        for o, d in self.od_pairs:
            self.served_demand[o, d] = defaultdict(float)
        self.pax_flow = defaultdict(dict)
        for o, d in self.od_pairs:
            self.pax_flow[o, d] = defaultdict(float)  # TODO: Or float? idk
        self.reb_flow = defaultdict(dict)
        for edge in self.network.edges(keys=True):
            self.reb_flow[edge] = defaultdict(float)

        # Vehicle count related variables
        self.acc = {
            node: [0 for i in range(self.total_time + self.extra_timesteps)]
            for node in self.network.nodes
            if node[-1] != "*"
        }  # number of vehicles within each node, acc[i][t] <-> number of vehicles at node i at time t
        self.dacc = defaultdict(
            lambda: defaultdict(int)
        )  # number of vehicles arriving at each node, dacc[i][t] <-> number of vehicles arriving at node i at time t
        for node in self.network.nodes:
            if node[-1] == "*":
                continue
            self.acc[node][0] = self.network.nodes[node][
                "accInit"
            ]  # No accInit in Sioux Falls network. Random generation instead.
            self.dacc[node] = defaultdict(float)

        # LTM related variables
        self.cvn = defaultdict(EdgeKeyDict)  # Cumulative Vehicle Number
        self.ffs = scenario.ffs
        self.cvn = {  # Initialize N(x, t) for all edges
            tuple(edge): {
                t: {0: 0, 1: 0}  # Upstream end  # Downstream end
                for t in range(self.total_time + 1)
            }
            for edge in self.network.edges
        }  # N(x, t)<->N[x][t][0/1]
        self.sending_flow = defaultdict(EdgeKeyDict)  # Sending Flow
        self.sending_flow = {  # Initialize sending flow for all edges
            tuple(edge): {t: 0 for t in range(self.total_time + 1)}
            for edge in self.network.edges
        }  # S_i(t) <-> sending_flow[i][t]
        self.receiving_flow = defaultdict(EdgeKeyDict)  # Receiving Flow
        self.receiving_flow = {  # Initialize receiving flow for all edges
            tuple(edge): {t: 0 for t in range(self.total_time + 1)}
            for edge in self.network.edges
        }  # R_i(t) <-> receiving_flow[i][t]
        self.node_transition_demand = defaultdict(
            EdgeKeyDict
        )  # Node Transition Demand, for storing demand of edge i to edge j, forall i in in_edges and j in out_edges
        self.node_transition_demand = (
            {  # Initialize node transition demand for all nodes
                node: {
                    i: {
                        j: {t: 0 for t in range(self.total_time + 1)}
                        for j in self.network.out_edges(node, keys=True)
                    }
                    for i in self.network.in_edges(node, keys=True)
                }
                for node in self.network.nodes
            }
        )
        self.pending_node_transition_demand = defaultdict(
            EdgeKeyDict
        )  # Node Transition Demand, for storing demand of edge i to edge j, forall i in in_edges and j in out_edges
        self.pending_node_transition_demand = (
            {  # Initialize node transition demand for all nodes
                node: {
                    i: {
                        j: {t: 0 for t in range(self.total_time + 1)}
                        for j in self.network.out_edges(node, keys=True)
                    }
                    for i in self.network.in_edges(node, keys=True)
                }
                for node in self.network.nodes
            }
        )
        self.link_mean_travel_time = defaultdict(EdgeKeyDict)
        self.link_mean_travel_time = {
            tuple(edge): {t: 0 for t in range(self.total_time + 1)}
            for edge in self.network.edges
        }
        self.link_traffic_flow = defaultdict(EdgeKeyDict)
        self.link_traffic_flow = {
            tuple(edge): [0 for t in range(self.total_time + 1)]
            for edge in self.network.edges
        }  # self.link_traffic_flow[edge][t] <-> traffic flow on edge at time t
        # Misc variables
        self.beta = beta * scenario.time_step
        self.info = dict.fromkeys(
            ["revenue", "served_demand", "rebalancing_cost", "operating_cost"], 0
        )
        self.reward = 0
        self.cache = PathCacheManager(self.network)
        self.cache.load_cache()

    def cache_paths(self):
        cache = PathCacheManager(self.network)
        cache.cache_paths(self.origins, self.destinations)

    def calculate_sending_flow(self, edge, edge_attrib, t):
        delta_t = self.time_step
        try:
            ffs = edge_attrib["ffs"]
        except KeyError:
            ffs = self.ffs  # Default ffs value
        # Source nodes and destination nodes don't have 'w' and 'k_j' attribute.
        if t + delta_t - 60 * edge_attrib["length"] / ffs < 0:  # length/ffs is hour.
            return min(
                self.cvn[tuple(edge)][0][0] - self.cvn[tuple(edge)][t][1],
                edge_attrib["q_max"],
            )
        return min(
            self.cvn[tuple(edge)][int(t + delta_t - 60 * edge_attrib["length"] / ffs)][
                0
            ]
            - self.cvn[tuple(edge)][t][1],
            edge_attrib["q_max"] * delta_t,
        )

    def calculate_receiving_flow(self, edge, edge_attrib, t):
        delta_t = self.time_step
        if edge_attrib["type"] == "destination" or edge_attrib["type"] == "origin":
            return np.inf
        try:
            return min(
                self.cvn[tuple(edge)][
                    int(t + delta_t + edge_attrib["length"] / edge_attrib["w"])
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

    def eval_network_gen_cost(self, time, coeffs):
        for idx, edge in enumerate(
            edges := [
                edge
                for edge in self.network.edges(keys=True)
                if edge[0][-1] != "*" and edge[1][-1] != "*"
            ]
        ):
            if time == 0:
                self.gen_cost[edge][time] = self.network.edges[edge]["free_flow_time"]
            else:
                self.gen_cost[edge][time] = self.approximate_gen_cost_function(
                    current_traffic_flow=self.link_traffic_flow[edge][time],
                    avg_traffic_flow=np.average(self.link_traffic_flow[edge][:time]),
                    coeffs=coeffs,
                    idx=idx,
                )

    def approximate_gen_cost_function(
        self, current_traffic_flow, avg_traffic_flow, coeffs, idx
    ):
        avg_traffic_flow = 0
        if coeffs is not None:
            coeff = coeffs[idx]
            approximation = coeff[0]

            # Add the rest of the Taylor series terms
            for i in range(1, len(coeff)):
                term = (
                    coeff[i]
                    * (current_traffic_flow - avg_traffic_flow) ** i
                    / math.factorial(i)
                )
                approximation += term
            if np.isnan(approximation):
                approximation = coeff[0]  # Fall back to first term if invalid
            return approximation
        else:
            return 1  # TODO: What is the default value?

    def update_traffic_flow_and_travel_time(self, time):
        for edge in self.network.edges(keys=True):
            if (
                self.network.edges[edge]["type"] == "origin"
                or self.network.edges[edge]["type"] == "destination"
            ):
                continue
            self.link_traffic_flow[edge][time] = self.get_link_traffic_flow(edge, time)
            self.link_mean_travel_time[edge][time] = self.get_link_travel_time(
                edge, time=time
            )

    def generate_path_ids(
        self,
        network,
        pax_demand,
        time,
        gen_cost,
        acc,
        origins,
        destinations,
        only_pickup_nearby=True,
    ):
        """
        Generate unique path IDs and their costs for IOD paths.

        Args:
        - network: NetworkX graph object representing the road network.
        - pax_demand: Dictionary of passenger demand at each time step.
        - time: Current time step.
        - gen_cost: Dictionary of generalized costs for each edge at each time step.
        - acc: Dictionary of vehicle counts at each node at each time step.
        - origins: List of origin nodes.
        - destinations: List of destination nodes.
        - only_pickup_nearby: Boolean flag to only consider nearby pickups. By nearby we mean only pick up pax at just one edge away from current location, as well as the current location itself.

        Returns:
        - iod_path_tuple: A list of tuples for IOD paths with path IDs and costs.
        - iod_path_dict: A dictionary of dictionaries for IOD paths with path IDs and costs.
        """
        # start = tm.time()
        path_id = 0
        iod_path_tuple = []
        iod_path_dict = {}  # iod_path_dict[(i,o,d)][path_id] = (path, cost)
        if self.cache is None:
            self.cache = PathCacheManager(self.network)
            self.cache.load_cache()
        cache = self.cache
        # Generate IOD paths with unique IDs
        if only_pickup_nearby:
            for o, d in pax_demand[time].keys():
                if o == d:
                    continue
                acc_candidates = [
                    i for i in self.network.predecessors(o) if i[-1] != "*"
                ]
                acc_candidates.append(o)
                for i in acc_candidates:
                    io_paths, od_paths = cache.get_cached_paths(i, o, d)
                    for io_path in io_paths:
                        for od_path in od_paths:
                            # Combine IO and OD paths, excluding the duplicate occurrence of 'o'
                            combined_path = io_path + od_path
                            # Calculate the total cost of the combined path
                            total_cost = sum(
                                gen_cost[edge][time]
                                for edge in combined_path
                                if not np.isnan(gen_cost[edge][time])
                            )
                            # Update tuples and dictionaries with the new path and its cost
                            if (
                                time in self.pax_demand
                                and (o, d) in self.pax_demand[time]
                                and self.pax_demand[time][(o, d)] != 0
                            ):
                                iod_path_tuple.append(
                                    (
                                        i,  # Vehicle current location
                                        o,  # Trip origin
                                        d,  # Trip destination
                                        path_id,  # Unique Path ID
                                        total_cost,  # Total cost of the path (generalized)
                                        self.pax_demand[time][(o, d)][0],  # Demand
                                        self.pax_demand[time][(o, d)][1],  # Price
                                    )
                                )
                            if (i, o, d) not in iod_path_dict:
                                iod_path_dict[(i, o, d)] = {}
                            iod_path_dict[path_id] = (
                                tuple(combined_path),
                                total_cost,
                                (i, o, d),
                            )
                            path_id += 1
        else:
            for i, o, d in product(self.acc.keys(), origins, destinations):
                if o == d:
                    continue
                io_paths, od_paths = cache.get_cached_paths(i, o, d)
                for io_path in io_paths:
                    for od_path in od_paths:
                        # Combine IO and OD paths, excluding the duplicate occurrence of 'o'
                        combined_path = io_path + od_path
                        # Calculate the total cost of the combined path
                        total_cost = sum(
                            [gen_cost[edge][time] for edge in combined_path]
                        )
                        # Update tuples and dictionaries with the new path and its cost
                        if (
                            time in self.pax_demand
                            and (o, d) in self.pax_demand[time]
                            and self.pax_demand[time][(o, d)] != 0
                        ):
                            iod_path_tuple.append(
                                (
                                    i,  # Vehicle current location
                                    o,  # Trip origin
                                    d,  # Trip destination
                                    path_id,  # Unique Path ID
                                    total_cost,  # Total cost of the path (generalized)
                                    self.pax_demand[time][(o, d)][0],  # Demand
                                    self.pax_demand[time][(o, d)][1],  # Price
                                )
                            )
                        if (i, o, d) not in iod_path_dict:
                            iod_path_dict[(i, o, d)] = {}
                        iod_path_dict[path_id] = (
                            tuple(combined_path),
                            total_cost,
                            (i, o, d),
                        )
                        path_id += 1
        # print(f"Time taken to generate IOD paths: {tm.time() - start} seconds.")
        return iod_path_tuple, iod_path_dict

    def format_for_opl(self, value):
        """Format Python data types into strings that OPL can parse."""
        if isinstance(value, list):
            return "{" + ", ".join(self.format_for_opl(v) for v in value) + "}"
        elif isinstance(value, tuple):
            return "<" + ", ".join(self.format_for_opl(v) for v in value) + ">"
        elif isinstance(value, dict):
            return (
                "{"
                + " ".join(f"{k}={self.format_for_opl(v)}" for k, v in value.items())
                + "}"
            )
        else:
            return str(value)

    def matching(self, CPLEXPATH=None, PATH="", platform="win"):
        t = self.time
        acc_tuple = [
            (i, self.acc[i][t + 1]) for i in self.acc
        ]  # Accumulation attributes
        iod_path_tuple, iod_path_dict = self.generate_path_ids(
            network=self.network,
            pax_demand=self.pax_demand,
            time=t,
            gen_cost=self.gen_cost,
            acc=self.acc,
            origins=self.origins,
            destinations=self.destinations,
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
            f.write('path="' + res_file + '";\r')
            # Writing accInitTuple
            f.write(f"accInitTuple = {self.format_for_opl(acc_tuple)};\n")
            # Writing iodPathTuple
            f.write(f"ioPathTuple = {self.format_for_opl(iod_path_tuple)};\n")
        mod_file = mod_path + "matching_path.mod"
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
        with open(res_file, "r", encoding="utf8") as file:
            for row in file:
                item = row.replace("e)", ")").strip().strip(";").split("=")
                if item[0] == "flow":
                    values = item[1].strip(")]").strip("[(").split(")(")
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, o, d, pid, f = v.split(",")
                        flow[str(i), str(o), str(d), int(pid)] = float(f)
        pax_action = {key: value for key, value in flow.items() if value > 1e-6}
        # pax_action: Retreived from OPL and formulated to be used in pax_step and LTM. pax_action[(i,o,d,path_id)] = flow starting from i to o to d, using path_id. For path_id correspondence see iod_path_tuple.
        # iod_path_dict: Formulated to be used in pax_step and LTM. iod_path_dict[path_id] = (path, cost)
        return pax_action, iod_path_dict

    def pax_step(
        self,
        paxAction=None,
        CPLEXPATH=None,
        PATH="",
        platform="win",
        taylor_params=None,
    ):
        t = self.time
        delta_t = self.time_step
        # Do a step in passenger matching
        for i in self.network.nodes:
            if i[-1] == "*":
                continue
            self.acc[i][t + 1] += self.acc[i][t]
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
            pid: flow for (_, _, _, pid), flow in paxAction.items() if flow > 1e-6
        }
        for pid in iod_path_demands.keys():
            self.path_demands[paxPathDict[pid][0]][t] = iod_path_demands[pid]
        # Obtain link demands from path demands
        # if self.link_demands is None:
        for pid, flow in iod_path_demands.items():
            self.path_demands[paxPathDict[pid][0]][t + delta_t] += flow
            for edge in self.network.in_edges(paxPathDict[pid][0][0][0], keys=True):
                if self.network.edges[edge]["type"] == "origin":
                    o_edge = edge
            for edge in self.network.out_edges(paxPathDict[pid][0][-1][1], keys=True):
                if self.network.edges[edge]["type"] == "destination":
                    d_edge = edge
            path = (o_edge,) + paxPathDict[pid][0] + (d_edge,)
            for i in range(len(path) - 1):
                self.node_transition_demand[path[i][1]][path[i]][path[i + 1]][t] += flow
            path_travel_time = 0
            for edge_start, edge_end, edge_key in paxPathDict[pid][0]:
                self.link_demands[(edge_start, edge_end, edge_key)][t] += flow
                path_travel_time += self.link_mean_travel_time[
                    (edge_start, edge_end, edge_key)
                ][t]
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
            i, o, d = (
                paxPathDict[pid][2][0],
                paxPathDict[pid][2][1],
                paxPathDict[pid][2][2],
            )
            assert iod_path_demands[pid] < self.acc[i][t + 1] + 1e-3
            self.served_demand[o, d][t] += iod_path_demands[pid]
            self.pax_flow[o, d][int(t + path_travel_time)] += iod_path_demands[pid]
            self.info["operating_cost"] += (
                path_travel_time * self.beta * iod_path_demands[pid]
            )
            self.acc[i][t + 1] -= iod_path_demands[pid]
            if int(t + path_travel_time) < self.total_time + self.extra_timesteps:
                self.acc[d][int(t + path_travel_time)] += iod_path_demands[pid]
            # print(path_travel_time)
            self.info["served_demand"] += self.served_demand[o, d][t]
            self.dacc[d][int(t + path_travel_time)] += self.pax_flow[o, d][
                int(t + path_travel_time)
            ]
            self.reward += iod_path_demands[pid] * (
                self.pax_demand[t][(o, d)][1] - self.beta * path_travel_time
            )
            self.info["revenue"] += (
                iod_path_demands[pid] * self.pax_demand[t][(o, d)][1]
            )

        # self.obs = (self.acc, self.time, self.dacc, self.pax_demand)
        self.obs = (
            self.acc,
            self.time,
            self.dacc,
            self.pax_demand,
            self.link_demands,
            self.link_traffic_flow,
        )
        done = False
        return self.obs, max(0, self.reward), done, self.info

    def reb_step(self, rebAction):
        t = self.time
        self.reward = 0
        self.rebAction = rebAction

        # Rebalancing
        if -1 in rebAction.values():
            # Model unbounded
            self.reward = -1e6
            self.obs = (self.acc, self.time, self.dacc, self.pax_demand)
            done = t == self.total_time
            self.info["rebalancing_cost"] += 1e6
            self.info["operating_cost"] += 1e6
            return self.obs, self.reward, done, self.info

        for idx, node in enumerate(
            [node for node in self.network.nodes if node[-1] != "*"]
        ):
            target_outflow = {
                edge: rebAction[edge]
                for edge in self.network.out_edges(node, keys=True)
                if edge[0][-1] != "*" and edge[1][-1] != "*"
            }
            total_outflow = sum(target_outflow.values())
            if total_outflow == 0:
                continue
            ratios = {
                edge: target_outflow[edge] / total_outflow
                for edge in target_outflow.keys()
            }
            if total_outflow > self.acc[node][t + 1]:
                total_outflow = self.acc[node][t + 1]
            self.acc[node][t + 1] -= total_outflow
            for idx, edge in enumerate(target_outflow.keys()):
                actual_outflow = ratios[edge] * total_outflow
                for in_edge in self.network.in_edges(edge[0], keys=True):
                    if self.network.edges[in_edge]["type"] == "origin":
                        self.link_demands[in_edge][t] += actual_outflow
                for out_edge in self.network.out_edges(edge[1], keys=True):
                    if self.network.edges[out_edge]["type"] == "destination":
                        self.link_demands[out_edge][t] += actual_outflow
                self.reb_flow[edge][
                    t + self.link_mean_travel_time[edge][t]
                ] = actual_outflow
                self.dacc[edge[1]][
                    t + self.link_mean_travel_time[edge][t]
                ] += actual_outflow
                self.info["rebalancing_cost"] += (
                    self.link_mean_travel_time[edge][t] * self.beta * actual_outflow
                )
                self.info["operating_cost"] += (
                    self.link_mean_travel_time[edge][t] * self.beta * actual_outflow
                )
                self.reward -= actual_outflow * self.link_mean_travel_time[edge][t]

        # arrival for the next time step, executed in the last state of a time step
        # this makes the code slightly different from the previous version, where the following codes are executed between matching and rebalancing
        for k, (edge_start, edge_end, edge_key) in enumerate(
            [
                edge
                for edge in self.network.edges(keys=True, data=False)
                if edge[0][-1] != "*" and edge[1][-1] != "*"
            ]
        ):
            if (
                (edge_start, edge_end, edge_key) in self.reb_flow
                and t in self.reb_flow[edge_start, edge_end, edge_key]
                and t + self.link_mean_travel_time[(edge_start, edge_end, edge_key)][t]
                < self.total_time + self.extra_timesteps
            ):
                self.acc[edge_end][
                    int(
                        t
                        + self.link_mean_travel_time[(edge_start, edge_end, edge_key)][
                            t
                        ]
                    )
                ] += self.reb_flow[edge_start, edge_end, edge_key][t]

        # self.time += 1
        self.obs = (self.acc, self.time, self.dacc, self.pax_demand)
        done = self.time == self.total_time
        return self.obs, self.reward, done, self.info

    def ltm_step(self, use_ctm_at_merge=False):
        t = self.time
        delta_t = self.time_step
        # Do a step in LTM
        for node in sorted(
            list(self.network.nodes),
            key=lambda x: (not x.endswith("*"), x.endswith("**")),
        ):
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
                        [
                            (
                                self.path_demands[path][t + delta_t]
                                if path[0][0] == out_node
                                else 0
                            )
                            for path in self.path_demands.keys()
                        ]
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
                # sum_inout = sum(
                #     [
                #         self.node_transition_demand[node][i][j][t]
                #         for i in in_edges
                #         for j in out_edges
                #     ]
                # )
                p = 1 / (len(in_edges) * len(out_edges))
                for in_edge in in_edges:
                    for out_edge in out_edges:
                        if p > 1e-6:
                            summ = sum([p * self.sending_flow[i][t] for i in in_edges])
                            if summ > 1e-6:
                                transition_flow = p * min(
                                    min(
                                        [
                                            (
                                                self.receiving_flow[out_edge][t]
                                                * self.sending_flow[in_edge][t]
                                                / summ
                                                if self.receiving_flow[out_edge][t]
                                                != np.inf
                                                else np.inf
                                            )
                                        ]
                                    ),
                                    self.sending_flow[in_edge][t],
                                )
                            else:
                                transition_flow = 0
                        else:
                            transition_flow = 0
                        self.cvn[out_edge][t + delta_t][0] = (
                            self.cvn[out_edge][t][0] + transition_flow
                        )
                        self.cvn[in_edge][t + delta_t][1] = (
                            self.cvn[in_edge][t][1] + transition_flow
                        )
        self.time += self.time_step

    def reset(self, scenario):
        self.__init__(scenario=scenario)

    def get_link_mean_travel_time(self, edge):
        inv_1 = np.interp(
            np.arange(self.cvn[edge][self.total_time - 1][0]),
            [self.cvn[edge][t][0] for t in range(self.total_time)],
            np.arange(self.total_time),
        )
        inv_2 = np.interp(
            np.arange(self.cvn[edge][self.total_time - 1][1]),
            [self.cvn[edge][t][1] for t in range(self.total_time)],
            np.arange(self.total_time),
        )
        return sum([inv_2[i] - inv_1[i] for i in range(len(inv_2))]) / len(inv_2)

    def get_link_travel_time(self, edge, time):
        if (
            self.network.edges[edge]["type"] == "origin"
            or self.network.edges[edge]["type"] == "destination"
        ):
            raise ValueError("Cannot calculate travel time for dummy links.")
        if self.cvn[edge][time][0] == 0:
            return self.network.edges[edge]["free_flow_time"]
        else:
            # # Initialize x to be the time step just before the given time
            # x = time - 1
            # # Loop backwards to find the x such that cvn[edge][x][0] == cvn[edge][time][1]
            # while x >= 0:
            #     if self.cvn[edge][x][0] == self.cvn[edge][time][1]:
            #         break  # Found the desired time x
            #     x -= 1
            # # If x is found within the valid range, calculate and return the travel time
            # if x >= 0:
            #     return time - x
            # else:
            #     raise ValueError("No suitable x found within the valid time range.")
            values = np.array(
                [self.cvn[edge][x][0] for x in range(len(self.cvn[edge]))]
            )
            n = np.argmin(np.abs(values - self.cvn[edge][time][1]))
            return time - n

    def get_link_traffic_flow(self, edge, time):
        return self.cvn[edge][time][0] - self.cvn[edge][time][1]


class Scenario:
    """
    Class for AMoD environment scenario. Loads network from json file or generates sample network.
    """

    def __init__(
        self,
        use_sample_network=True,
        sample_network_name="sioux_falls",
        sd=None,
        total_time=60,
        time_step=1,
        json_file=None,
        json_hr=9,
        json_tstep=2,
        json_regions=None,
        ffs=60,
    ):
        """
        `Scenario` class for AMoD environment. Does all the network loading/generating from sample network/json file.

        Parameters:

        `use_sample_network`: bool, whether to use the sample network or not

        `sample_network_name`: str, name of the sample network

        `seed`: int, seed for random number generator

        `total_time`: int, total time of simulation

        `json_file`: str, path to json file

        `json_hr`: int, hour of the day in json file

        `json_tstep`: int, time step in json file

        `ffs`: float, free flow speed of network
        """
        self.seed = sd
        if sd is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        self.total_time = total_time
        self.time_step = time_step
        self.nedges = 0
        self.nregions = 0
        self.demand_input = defaultdict(dict)
        self.price = defaultdict(dict)
        if use_sample_network:
            self.is_json = False
            (
                self.G,
                self.pax_demand,
                self.origins,
                self.destinations,
                self.od_pairs,
                self.nregions,
                self.nedges,
            ) = self._load_sample_network(sample_network_name)
        else:
            # If Using json file. TODO: Need a json file to complete.
            raise NotImplementedError
        self.initial_travel_time = {
            edge: self.G.edges[edge]["length"] / ffs for edge in self.G.edges(keys=True)
        }
        self.ffs = ffs

    def _load_sample_network(self, name, demand_scale=(0, 5), price_scale=(10, 30)):
        """
        Loads sample network.
        """
        G = nx.MultiDiGraph()
        if name == "sample":
            origins = set()
            destinations = set()
            od_pairs = set()
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
            origins.add("A")
            destinations.add("D")
            od_pairs.add(("A", "D"))
            G = self._generate_dummy_od_links(G, self.origins, self.destinations)

            pax_demand = {
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
        elif name == "sioux_falls":
            file_path_network = "TransportationNetworks\SiouxFalls\SiouxFalls_net.tntp"
            file_path_trips = "TransportationNetworks\SiouxFalls\SiouxFalls_trips.tntp"

            pax_demand = self._initialize_pax_demand()

            edges = self._read_network_sioux_falls(file_path_network)
            G = self._create_multidigraph(edges)
            G = self._generate_init_acc(G)
            nregions = len(G.nodes)
            nedges = len(G.edges)
            base_demand, origins, destinations, od_pairs = (
                self._read_demand_sioux_falls(file_path_trips)
            )
            pax_demand = self._distribute_temporal_demand(base_demand, price_scale)
            G = self._generate_dummy_od_links(G, origins, destinations)
        else:
            raise ValueError(f"Unsupported network name: {name}")

        return G, pax_demand, origins, destinations, od_pairs, nregions, nedges

    def _initialize_pax_demand(self):
        # Initialize the passenger demand dictionary for each time step
        return {time: {} for time in range(60)}

    def _read_network_sioux_falls(self, file_path, backward_speed=10):
        with open(file_path, "r") as file:
            lines = file.readlines()

        start_index = 0
        for i, line in enumerate(lines):
            if "init_node" in line:
                start_index = i + 1
                break

        edges = []
        for line in lines[start_index:]:
            if line.strip():
                parts = line.split()
                edge_data = {
                    "init_node": str(parts[0]),
                    "term_node": str(parts[1]),
                    "capacity": float(parts[2]),
                    "length": float(parts[3]),  # In km
                    "free_flow_time": float(parts[4]),  # In minutes
                    "b": float(parts[5]),
                    "power": float(parts[6]),
                    "speed_limit": float(parts[7]),
                    "toll": float(parts[8]),
                    "type": "normal",  # Read edges are all normal, will generate dummy edges later, and their type would be "origin" or "destination"
                    "link_type": int(parts[9]),
                    "ffs": float(parts[3])
                    / (float(parts[4]) / 60),  # free flow speed, in km/h
                    "q_max": float(parts[2]),  # capacity (flow)
                    "w": backward_speed,  # backward speed, here is is arbitrary
                    "k_j": float(parts[2]) / (float(parts[3]) / (float(parts[4]) / 60))
                    + float(parts[2])
                    / backward_speed,  # jam density (calculated thru backward speed and TFD)
                }
                edges.append(edge_data)
        return edges

    def _create_multidigraph(self, edges):
        G = nx.MultiDiGraph()
        for edge in edges:
            G.add_edge(edge["init_node"], edge["term_node"], **edge)
        return G

    def _read_demand_sioux_falls(self, file_path):
        origins = set()
        destinations = set()
        od_pairs = set()
        with open(file_path, "r") as file:
            lines = file.readlines()

        demand = {}
        origin = None
        for line in lines:
            if line.startswith("Origin"):
                parts = line.split()
                origin = str(parts[1])
                origins.add(origin)
                demand[origin] = {}
            elif origin is not None:
                parts = line.strip().split(";")
                for part in parts:
                    if ":" in part:
                        dest, flow = part.split(":")
                        origins.add(origin)
                        destinations.add(str(dest.strip()))
                        od_pairs.add((origin, str(dest.strip())))
                        demand[origin][str(dest.strip())] = float(flow.strip())
        return demand, origins, destinations, od_pairs

    def _distribute_temporal_demand(
        self, base_demand, price_scale, peak_time=30, std_dev=20, scale=0.2, demand_scale=0.1
    ):
        time_periods = range(60)  # Time steps from 0 to 59
        pax_demand = {time: {} for time in time_periods}
        for time in time_periods:
            # Example temporal distribution, adjust as needed
            time_factor = self._temporal_distribution_factor(
                time, peak_time=peak_time, std_dev=std_dev
            )
            for origin, destinations in base_demand.items():
                for destination, od_demand in destinations.items():
                    if random.random() < scale:
                        adjusted_demand = od_demand * time_factor
                        price = random.randint(price_scale[0], price_scale[1])
                        pax_demand[time][(origin, destination)] = (
                            adjusted_demand / demand_scale,
                            price,
                        )
                        self.demand_input[origin, destination][time] = adjusted_demand / demand_scale
                        self.price[origin, destination][time] = price
        return pax_demand

    def _temporal_distribution_factor(self, time, peak_time, std_dev):
        # Example using a simple normal distribution for temporal variation
        return np.exp(-0.5 * ((time - peak_time) / std_dev) ** 2)

    def _generate_init_acc(self, network, acc_scale: tuple = (0, 200)):
        accs = {
            i: {"accInit": random.randint(acc_scale[0], acc_scale[1])}
            for i in network.nodes
        }
        nx.set_node_attributes(network, accs)
        return network

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
            network.add_node(d + "**")
            network.add_edge(
                d,
                d + "**",
                length=dummy_length,
                q_max=dummy_q_max,
                k_j=dummy_k_j,
                w=dummy_w,
                type="destination",
            )
        return network
