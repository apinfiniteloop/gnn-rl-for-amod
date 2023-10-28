import networkx as nx
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

ffs = 0.2 # free flow speed
delta_t = 1 # time step
# EPS = 1e-6 # epsilon for numerical stability


# For storing dicts with edge as key
class EdgeKeyDict(dict):
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = tuple(key)
        return super().__getitem__(key)

class LTM:
    # Link Transmission Model, only consider one O-D pair, and 2 paths. 
    def __init__(self, network, demand, paths, total_time):
        self.network = network
        self.path_demands = demand[0]
        self.link_demands = demand[1]
        self.paths = paths
        self.total_time = total_time
        self.N = defaultdict(EdgeKeyDict) # Cumulative Vehicle Number
        self.N = { #Initialize N(x, t) for all edges
            tuple(edge): {
                t:{
                    0: 0, # Upstream end
                    1: 0  # Downstream end
                } for t in range(total_time)
            } for edge in self.network.edges
        } # N(x, t)<->N[x][t][0/1]
        self.sending_flow = defaultdict(EdgeKeyDict) # Sending Flow
        self.sending_flow = { #Initialize sending flow for all edges
            tuple(edge): {
                t: 0 for t in range(total_time)
            } for edge in self.network.edges
        } # S_i(t) <-> sending_flow[i][t]
        self.receiving_flow = defaultdict(EdgeKeyDict) # Receiving Flow
        self.receiving_flow = { #Initialize receiving flow for all edges
            tuple(edge): {
                t: 0 for t in range(total_time)
            } for edge in self.network.edges
        } # R_i(t) <-> receiving_flow[i][t]
        # self.edge_demand = defaultdict(EdgeKeyDict) # Edge Demand
        # self.edge_demand = { #Initialize edge demand for all edges
        #     tuple(edge): {
        #         t: 0 for t in range(total_time)
        #     } for edge in self.network.edges
        # }
        self.node_transition_demand = defaultdict(EdgeKeyDict) # Node Transition Demand, for storing demand of edge i to edge j, forall i in in_edges and j in out_edges
        self.node_transition_demand = { #Initialize node transition demand for all nodes
            node: {
                i: {
                    j: {
                        t: 0 for t in range(total_time)
                    } for j in self.network.out_edges(node, keys=True)
                } for i in self.network.in_edges(node, keys=True)
            } for node in self.network.nodes
        }
        # Calculate node transition demand based on path demand
        for path_id, path in enumerate(paths):
            for time in range(total_time):
                for i in range(len(path)-1):
                    self.node_transition_demand[path[i][1]][path[i]][path[i+1]][time] += self.path_demands[path_id][time]

    def calculate_sending_flow(self, edge, edge_attrib, t):
        # Source nodes and destination nodes don't have 'w' and 'k_j' attribute.
        if t+delta_t-edge_attrib['length']/ffs < 0:
            return min(self.N[tuple(edge)][0][0]-self.N[tuple(edge)][t][1], edge_attrib['q_max'])
        else:
            return min(self.N[tuple(edge)][t+delta_t-edge_attrib['length']/ffs][0]-self.N[tuple(edge)][t][1], edge_attrib['q_max']*delta_t)

    def calculate_receiving_flow(self, edge, edge_attrib, t):
        if edge_attrib['type'] == 'destination':
            # return min(self.N[tuple(edge)][t+delta_t][1]+edge_attrib['length']/edge_attrib['w']-self.N[tuple(edge)][t][0], edge_attrib['q_max'])
            return np.inf
        else:
            try:
                return min(self.N[tuple(edge)][t+delta_t+edge_attrib['length']/edge_attrib['w']][1]+edge_attrib['k_j']*edge_attrib['length']-self.N[tuple(edge)][t][0], edge_attrib['q_max']*delta_t)
            except KeyError:
                return min(self.N[tuple(edge)][total_time-1][1]+edge_attrib['k_j']*edge_attrib['length']-self.N[tuple(edge)][t][0], edge_attrib['q_max']*delta_t)

    def disaggregate_demand(self, t):
        return 0

    def simulate(self):
        for edge in self.network.edges(keys=True):
            if self.network.edges[edge]['type'] == 'origin':
                self.N[edge][0][0] = self.link_demands[edge][0]
        for t in range(self.total_time-1):
            for node in self.network.nodes:
                # For each node, calculate the sending flow and receiving flow of its connected edges
                for edge in set(out_edges:=self.network.out_edges(nbunch=node, keys=True)) | set(in_edges:=self.network.in_edges(nbunch=node, keys=True)):
                    self.sending_flow[tuple(edge)][t] = self.calculate_sending_flow(edge, self.network.edges[edge], t)
                    self.receiving_flow[tuple(edge)][t] = self.calculate_receiving_flow(edge, self.network.edges[edge], t)
                # # Determine node's incoming and outgoing edges count
                # in_edges = self.network.in_edges(node, keys=True)
                # out_edges = self.network.out_edges(node, keys=True)
                # If is origin node, update N(x, t) for outgoing edges
                if len(in_edges) == 0 and len(out_edges) == 1:
                    transition_flow = min(sum([self.path_demands[i][t+delta_t] for i in self.path_demands.keys()]), self.receiving_flow[tuple(out_edges)[0]][t])
                    self.N[tuple(out_edges)[0]][t+delta_t][0] = self.N[tuple(out_edges)[0]][t][0] + transition_flow
                # If is homogenous node, update N(x, t) for outgoing edges
                elif len(in_edges) == 1 and len(out_edges) == 1:
                    transition_flow = min(self.sending_flow[tuple(in_edges)[0]][t], self.receiving_flow[tuple(out_edges)[0]][t])
                    self.N[tuple(in_edges)[0]][t+delta_t][1] = self.N[tuple(in_edges)[0]][t][1] + transition_flow
                    self.N[tuple(out_edges)[0]][t+delta_t][0] = self.N[tuple(out_edges)[0]][t][0] + transition_flow
                # If is split node, update N(x, t) for outgoing edges
                elif len(in_edges) == 1 and len(out_edges) > 1:
                    sum_transition_flow = 0
                    for edge in out_edges:
                        # self.disaggregate_demand(edge, t)
                        # if self.link_demands[tuple(in_edges)[0]][t] == 0:
                        #     continue
                        try:
                            p=self.link_demands[edge][t]/self.link_demands[tuple(in_edges)[0]][t]
                            transition_flow = p*min(self.sending_flow[tuple(in_edges)[0]][t], min([self.receiving_flow[e][t]/(self.link_demands[e][t]/self.link_demands[tuple(in_edges)[0]][t]) for e in out_edges]))
                        except ZeroDivisionError:
                            transition_flow = 0
                        self.N[tuple(edge)][t+delta_t][0] = self.N[tuple(edge)][t][0] + round(transition_flow)
                        sum_transition_flow += transition_flow
                    self.N[tuple(in_edges)[0]][t+delta_t][1] = self.N[tuple(in_edges)[0]][t][1] + round(sum_transition_flow)
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
                            p=self.link_demands[edge][t]/self.link_demands[tuple(out_edges)[0]][t]
                            transition_flow = min(self.sending_flow[edge][t], self.receiving_flow[tuple(out_edges)[0]][t]*self.sending_flow[edge][t]/(sum([self.sending_flow[e][t] for e in in_edges])))
                        except ZeroDivisionError:
                            transition_flow = 0
                        self.N[edge][t+delta_t][1] = self.N[edge][t][1] + transition_flow
                        sum_transition_flow += transition_flow
                    self.N[tuple(out_edges)[0]][t+delta_t][0] = self.N[tuple(out_edges)[0]][t][0] + sum_transition_flow
                # If is destination node, update N(x, t) for outgoing edges
                elif len(in_edges) == 1 and len(out_edges) == 0:
                    transition_flow = self.sending_flow[tuple(in_edges)[0]][t]
                    self.N[tuple(in_edges)[0]][t+delta_t][1] = self.N[tuple(in_edges)[0]][t][1] + transition_flow
                # If is normal node, update N(x, t) for outgoing edges
                elif len(in_edges) > 1 and len(out_edges) > 1:
                    sum_inout = sum([self.node_transition_demand[node][i][j][t] for i in in_edges for j in out_edges])
                    for in_edge in in_edges:
                        for out_edge in out_edges:
                            p=self.node_transition_demand[node][in_edge][out_edge][t]/sum_inout
                            transition_flow = p*min(min([self.receiving_flow[out_edge][t]*self.sending_flow[in_edge][t]/(sum([self.node_transition_demand[node][i][out_edge]*self.sending_flow[i][t] for i in in_edges]))]), self.sending_flow[in_edge][t])
                            self.N[out_edge][t+delta_t][0] = self.N[out_edge][t][0] + transition_flow
                            self.N[in_edge][t+delta_t][1] = self.N[in_edge][t][1] + transition_flow
    
    def plot_N(self, edge):
        plt.scatter([t for t in range(self.total_time)], [self.N[edge][t][0] for t in range(self.total_time)], label='Upstream end')
        plt.scatter([t for t in range(self.total_time)],[self.N[edge][t][1] for t in range(self.total_time)], label='Downstream end')
        plt.title("N(x, t) for edge {}".format(edge))
        plt.xlabel("Time")
        plt.ylabel("N(x, t)")
        plt.legend()
        plt.show()
    
    def get_link_travel_time(self, edge, t):
        return 0



def create_sample_network():
    G = nx.MultiDiGraph()

    G.add_node("Or")
    G.add_node("A")
    G.add_node("B")
    G.add_node("C")
    G.add_node("D")
    G.add_node("De")

    # G.add_edge("Or", "A", length=np.inf, q_max=np.inf, type='origin') # k_j, w are 0 now, but is debatable.
    G.add_edge("Or", "A", length=0, q_max=np.inf, k_j=np.inf, w=1e-6, type='origin')
    G.add_edge("A", "B", length=0.4, q_max=50, k_j=1000, w=0.1, type='normal')
    G.add_edge("B", "C", length=0.4, q_max=50, k_j=300, w=0.1, type='normal')
    G.add_edge("C", "D", length=0.8, q_max=50, k_j=60, w=0.1, type='normal')
    G.add_edge("C", "D", length=0.4, q_max=50, k_j=30, w=0.1, type='normal')
    # G.add_edge("D", "De", length=np.inf, q_max=np.inf, type='destination') # Same as above
    G.add_edge("D", "De", length=0, q_max=np.inf, k_j=np.inf, w=1e-6, type='destination')

    return G

def generate_sample_demand(network, total_time, time_step=1):
    origin = "A"
    destination = "D"
    all_paths = list(nx.all_simple_edge_paths(network, source=origin, target=destination))

    # A very simple demand generation function, considering only one O-D pair, and 2 paths. Need to revise later for more complex situations.
    path_demands = {path_id: [random.randint(0, 5) for _ in range(total_time // time_step)] for path_id, _ in enumerate(all_paths)}
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

if __name__ == "__main__":
    network = create_sample_network()   

    total_time = 600
    demand, paths = generate_sample_demand(network, total_time)
    # for edge in network.edges:
    #     print(tuple(edge))
    ltm = LTM(network, demand, paths, total_time)
    ltm.simulate()
    ltm.plot_N(('C', 'D', 1))
    # print(ltm.N)