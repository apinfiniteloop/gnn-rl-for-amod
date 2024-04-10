// Revised OPL Script for Passenger Matching Problem using numeric IDs

tuple iodPathTuple{
    int i;
    int o;
    int d;
    int id;
    float c;
    float de;
    float p;
}

tuple Accumulation {
    int i;
    float acc;
}

tuple iod_path {
    int i;
    int o;
    int d;
    int id;
}

tuple iod_Edge{
    int i; // Node ID
    int o; // Origin ID
    int d; // Destination ID
}


string path = ...;
{iodPathTuple} ioPathTuple = ...;
{Accumulation} accInitTuple = ...;
{iod_Edge} demandEdge = {<i,o,d>|<i,o,d,id,c,de,p> in ioPathTuple};
{iod_path} demandPath = {<i,o,d,id>|<i,o,d,id,c,de,p> in ioPathTuple};
{int} region = {i|<i,v> in accInitTuple};
float accInit[region] = [i:v|<i,v> in accInitTuple];
float demand[demandPath] = [<i,o,d,id>:de|<i,o,d,id,c,de,p> in ioPathTuple];
float price[demandPath] = [<i,o,d,id>:p|<i,o,d,id,c,de,p> in ioPathTuple];
float cost[demandPath] = [<i,o,d,id>:c|<i,o,d,id,c,de,p> in ioPathTuple];
dvar float+ demandFlow[demandPath];
maximize(sum(e in demandPath) demandFlow[e]*(price[e]-cost[e]));
subject to
{
    forall(i in region)
        0<= accInit[i] - sum(e in demandPath: e.i==i) demandFlow[e];

    forall(j in demandPath)
        sum(e in demandPath: e.o==j.o && e.d==j.d) demandFlow[e] <= demand[j];
}

main {
    thisOplModel.generate();
    cplex.solve();
    var ofile = new IloOplOutputFile(thisOplModel.path);
    ofile.write("flow=[")
    for(var e in thisOplModel.demandPath){
        ofile.write("(");
        ofile.write(e.i);
        ofile.write(",");
        ofile.write(e.o);
        ofile.write(",");
        ofile.write(e.d);
        ofile.write(",");
        ofile.write(e.id);
        ofile.write(",");
        ofile.write(thisOplModel.demandFlow[e]);
        ofile.writeln(")");
    }
    ofile.writeln("];");
    ofile.close();
}