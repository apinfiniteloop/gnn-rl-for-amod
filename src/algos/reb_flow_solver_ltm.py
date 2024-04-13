import os
import subprocess
from collections import defaultdict
from src.misc.utils import mat2str


def solveRebFlow(env, res_path, desiredAcc, gen_cost, CPLEXPATH):
    t = env.time
    accRLTuple = [(int(n), int(round(desiredAcc[n]))) for n in desiredAcc]
    accTuple = [(int(n), int(env.acc[n][t + 1])) for n in env.acc]
    edgeAttr = [
        (int(i), int(j), int(k), gen_cost[i, j, k][t] if (i, j, k) in gen_cost else 0)
        for i, j, k in env.network.edges(data=False, keys=True)
        if i[-1] != "*" and j[-1] != "*"
    ]
    modPath = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
    OPTPath = (
        os.getcwd().replace("\\", "/")
        + "/saved_files/cplex_logs/rebalancing/"
        + res_path
        + "/"
    )
    if not os.path.exists(OPTPath):
        os.makedirs(OPTPath)
    datafile = OPTPath + f"data_{t}.dat"
    resfile = OPTPath + f"res_{t}.dat"
    with open(datafile, "w") as file:
        file.write('path="' + resfile + '";\r\n')
        file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
        file.write("accInitTuple=" + mat2str(accTuple) + ";\r\n")
        file.write("accRLTuple=" + mat2str(accRLTuple) + ";\r\n")
    modfile = modPath + "minRebDistRebOnly_path.mod"
    if CPLEXPATH is None:
        CPLEXPATH = "/opt/ibm/ILOG/CPLEX_Studio128/opl/bin/x86-64_linux/"
    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
    out_file = OPTPath + f"out_{t}.dat"
    with open(out_file, "w") as output_f:
        subprocess.check_call(
            [CPLEXPATH + "oplrun", modfile, datafile], stdout=output_f, env=my_env
        )
    output_f.close()

    # 3. collect results from file
    flow = defaultdict(float)
    with open(resfile, "r", encoding="utf8") as file:
        for row in file:
            item = row.strip().strip(";").split("=")
            if item[0] == "flow":
                values = item[1].strip(")]").strip("[(").split(")(")
                for v in values:
                    if len(v) == 0:
                        continue
                    i, j, k, f = v.split(",")
                    flow[i, j, k] = float(f)
    # action = [
    #     flow[i, j, k]
    #     for i, j, k in env.network.edges(data=False, keys=True)
    #     if (i, j, k) in flow.keys()
    # ]
    return flow
