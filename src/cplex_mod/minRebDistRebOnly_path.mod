tuple Edge{
  int i;
  int j;
  int id;
}

tuple edgeAttrTuple{
    int i;
    int j;
    int id;
    float t;
}

tuple accTuple{
  int i;
  float n;
}

string path = ...;

{edgeAttrTuple} edgeAttr = ...;
{accTuple} accInitTuple = ...;
{accTuple} accRLTuple = ...;

{Edge} edges = {<i,j,id>|<i,j,id,t> in edgeAttr};
{int} region = {i|<i,v> in accInitTuple};

float time[edges] = [<i,j,id>:t|<i,j,id,t> in edgeAttr]; // TODO: distance --> we have no distance (replace with time?)
float desiredVehicles[region] = [i:v|<i,v> in accRLTuple]; // TODO: desiredVehicles
//float accInit[region] = [i:v|<i,v> in accInitTuple];
float vehicles[region] = [i:v|<i,v> in accInitTuple]; // TODO: vehicles

dvar float+ slack[region];
dvar int+ rebFlow[edges];

minimize(sum(e in edges) (rebFlow[e]*time[e])) + 0.05 * sum(i in region) slack[i];;
subject to
{
  forall(i in region)
    {
    A1:sum(e in edges: e.i==i && e.i!=e.j) (rebFlow[<e.j, e.i, e.id>] - rebFlow[<e.i, e.j, e.id>]) + slack[i] >= desiredVehicles[i] - vehicles[i];
    sum(e in edges: e.i==i && e.i!=e.j) rebFlow[<e.i, e.j, e.id>] <= vehicles[i];
    }
}

main {
  thisOplModel.generate();
  cplex.solve();
  var ofile = new IloOplOutputFile(thisOplModel.path);
  ofile.write("flow=[")
  for(var e in thisOplModel.edges)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(e.id);
          ofile.write(",");
         ofile.write(thisOplModel.rebFlow[e]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.close();
}