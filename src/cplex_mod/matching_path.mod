// Revised OPL Script for Passenger Matching Problem using numeric IDs

tuple iodPathTuple{
    key int i; // Node ID
    key int o; // Origin ID
    key int d; // Destination ID
    key int id;
    float c; // Cost
    float de; // Demand
    float p; // Price
}

tuple Accumulation {
    key int i; // Node ID
    float acc; // Accumulation at node
}

tuple iod_path {
    key int i;
    key int o;
    key int d;
    key int id;
}

tuple iod_Edge{
    key int i; // Node ID
    key int o; // Origin ID
    key int d; // Destination ID
}


string path = ...;
{iodPathTuple} ioPathTuple = ...; // Mapping from (i,o,d) to paths and costs
{Accumulation} accInitTuple = ...; // Initial accumulation at each node 

{iod_Edge} demandEdge = {<i,o,d>|<i,o,d,id,c,de,p> in  ioPathTuple}
{iod_path} demandPath = {<i,o,d,id>|<i,o,d,id,c,de,p> in ioPathTuple}
{int} region = {i|<i,v> in accInitTuple}
float accInit[region] = [i:v|<i,v> in accInitTuple]
float demand[demandPath] = [<i,o,d,id>:de|<i,o,d,id,c,de,p> in ioPathTuple]
float price[demandPath] = [<i,o,d,id>:p|<i,o,d,id,c,de,p> in ioPathTuple]
float cost[demandPath] = [<i,o,d,id>:c|<i,o,d,id,c,de,p> in ioPathTuple]
dvar float+ demandFlow[demandPath];
maximize(sum(i in demandPath) sum(o in demandPath) sum(d in demandPath) demandFlow[i,o,d]*(price[i,o,d]-cost[i,o,d])); //Needs revision
subject to
{
    forall(i in region)
        0<= accInit[i] - sum(o in demandPath) sum(d in demandPath) sum(id in demandPath) demandFlow[i,o,d,id];

    forall(e in demandEdge)
        demandFlow[e] <= demand[e];
}


// {int} V; // Set of node IDs
// {int} ODpairs; // Set of OD pair IDs, derived from demandAttr
// {PathCost} ioPathTuple[<int,int>]; // Mapping from (i,o) to paths and costs
// {PathCost} odPathTuple[<int,int>]; // Mapping from (o,d) to paths and costs

// // Decision Variables
// dvar boolean x[V][ODpairs][*][*]; // x[i][od][p1][p2] indicates whether a match is made

// // Objective: Maximize total profit
// maximize
//   sum(i in V, od in ODpairs, p1 in ioPathTuple[(i, demandAttr[od].o)], p2 in odPathTuple[(demandAttr[od].o, demandAttr[od].d)]) 
//     x[i][od][p1.id][p2.id] * (demandAttr[od].price - p1.cost - p2.cost);
    
// // Constraints
// subject to {
  
//   // Demand constraint for each OD pair
//   forall(od in ODpairs)
//     0 < sum(i in V, p1 in ioPathTuple[(i, demandAttr[od].o)], p2 in odPathTuple[(demandAttr[od].o, demandAttr[od].d)]) 
//       x[i][od][p1.id][p2.id] <= demandAttr[od].demand;
    
//   // Accumulation constraint at each node
//   forall(i in V)
//     0 < sum(od in ODpairs, p1 in ioPathTuple[(i, demandAttr[od].o)], p2 in odPathTuple[(demandAttr[od].o, demandAttr[od].d)]) 
//       x[i][od][p1.id][p2.id] <= accInitTuple[i].acc;
    
// }
