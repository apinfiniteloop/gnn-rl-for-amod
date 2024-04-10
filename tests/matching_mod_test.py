import os
from collections import defaultdict
import subprocess

t = 0
CPLEXPATH = "C:/Program Files/IBM/ILOG/CPLEX_Studio2211/opl/bin/x64_win64/"
mod_path = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
mod_file = mod_path + "matching_path.mod"
PATH = "scenario_nyc4"
matching_path = (
    os.getcwd().replace("\\", "/") + "/saved_files/cplex_logs/matching/" + PATH + "/"
)
platform = "win"
if CPLEXPATH is None:
    CPLEXPATH = "C:/Program Files/ibm/ILOG/CPLEX_Studio1210/opl/bin/x64_win64/"
my_env = os.environ.copy()
if platform == "mac":
    my_env["DYLD_LIBRARY_PATH"] = CPLEXPATH
else:
    my_env["LD_LIBRARY_PATH"] = CPLEXPATH
out_file = matching_path + "out_{}.dat".format(0)
data_file = matching_path + "data_{}.dat".format(t)
res_file = matching_path + "res_{}.dat".format(t)
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
                flow[int(i), int(o), int(d), int(pid)] = float(f)
