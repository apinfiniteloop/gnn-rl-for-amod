// Revised OPL Script for Passenger Matching Problem using numeric IDs

tuple Demand {
    key int o; // Origin ID
    key int d; // Destination ID
    float demand;
    float price;
}

tuple Accumulation {
    key int i; // Node ID
    float acc; // Accumulation at node
}

tuple PathCost {
    int id; // Path ID
    float cost; // Generalized cost of the path
}

// Declaring the sets
{Demand} demandAttr = ...;
{Accumulation} accInitTuple = ...;
{int} V; // Set of node IDs
{int} ODpairs; // Set of OD pair IDs, derived from demandAttr
{PathCost} ioPathTuple[<int,int>]; // Mapping from (i,o) to paths and costs
{PathCost} odPathTuple[<int,int>]; // Mapping from (o,d) to paths and costs

// Decision Variables
dvar boolean x[V][ODpairs][*][*]; // x[i][od][p1][p2] indicates whether a match is made

// Objective: Maximize total profit
maximize
  sum(i in V, od in ODpairs, p1 in ioPathTuple[(i, demandAttr[od].o)], p2 in odPathTuple[(demandAttr[od].o, demandAttr[od].d)]) 
    x[i][od][p1.id][p2.id] * (demandAttr[od].price - p1.cost - p2.cost);
    
// Constraints
subject to {
  
  // Demand constraint for each OD pair
  forall(od in ODpairs)
    0 < sum(i in V, p1 in ioPathTuple[(i, demandAttr[od].o)], p2 in odPathTuple[(demandAttr[od].o, demandAttr[od].d)]) 
      x[i][od][p1.id][p2.id] <= demandAttr[od].demand;
    
  // Accumulation constraint at each node
  forall(i in V)
    0 < sum(od in ODpairs, p1 in ioPathTuple[(i, demandAttr[od].o)], p2 in odPathTuple[(demandAttr[od].o, demandAttr[od].d)]) 
      x[i][od][p1.id][p2.id] <= accInitTuple[i].acc;
    
}
