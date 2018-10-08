import numpy as np
import math
from joblib import Parallel, delayed
from RHdata import RHinstance
from FHmodelsim import FhModelSim
from FHmodel import FhModel
from datetime import datetime
from MFHmodel import MFhModel

def get_fhbound(aRH, aT, ab, ad):
    afh = FhModel(aRH, aT, ab, ad)
    ub, _ = afh.solve()
    return ub

def get_fhbound_lower(aRH, aT, ab, ad):
    afh = FhModel(aRH, aT, ab, ad)
    lb = afh.solve_lower()
    return lb

def get_fhbound_lower_2(aRH, aT, ab, ad):
    afh = FhModel(aRH, aT, ab, ad)
    lb = afh.solve_lower_2()
    return lb

def get_mfhbound_lower(aRH, aT, ab, ad):
    if aT == 1:
        return "Infeasible"
    amfh = MFhModel(aRH, aT, ab, ad)
    lb = amfh.solve()
    return lb


n_samples = 10
beta = 0.95
n_samplesteps = int(math.ceil(-6.0 / np.log10(beta)))
load_samples = [0.6, 1.0, 1.2, 1.5, 2.0, 3.0, 6.0]
T_max = 50
mcT_max = range(1, T_max + 1)


# Get Sample Average Bounds
def run_bounds():
    outfile = open("results/sabounds.txt", "w")
    for j in range(len(load_samples)):
        rhinstance = RHinstance(beta, load_samples[j])
        alpha_b, alpha_d = rhinstance.getsamples(n_samples)
        for t in mcT_max:
            ub, lb1, lb2, lb3 = [], [], [], []
            ub = Parallel(n_jobs=-1, verbose=0)(delayed(get_fhbound)(rhinstance, t, alpha_b[sp], alpha_d[sp]) for sp in range(n_samples))
            lb1 = Parallel(n_jobs=-1, verbose=0)(delayed(get_fhbound_lower)(rhinstance, t, alpha_b[sp], alpha_d[sp]) for sp in range(n_samples))
            lb2 = Parallel(n_jobs=-1, verbose=0)(delayed(get_mfhbound_lower)(rhinstance, t, alpha_b[sp], alpha_d[sp]) for sp in range(n_samples))
            lb3 = Parallel(n_jobs=-1, verbose=0)(delayed(get_fhbound_lower_2)(rhinstance, t, alpha_b[sp], alpha_d[sp]) for sp in range(n_samples))
            ub_mean = int(np.mean(ub))
            lb1_mean = int(np.mean(lb1))
            if "Infeasible" in lb2:
                lb2_mean = "Infeasible"
            else:
                lb2_mean = int(np.mean(lb2))
            lb3_mean = int(np.mean(lb3))
            outStr = str(beta) + ',  ' +  str(load_samples[j]) + ',  ' + str(t) + ',  ' + str(ub_mean) + ',  ' + str(lb1_mean) + ',  ' + str(lb2_mean) + ',  ' + str(lb3_mean)
            print(outStr)
            print(outStr, file=outfile)
    outfile.close()


if __name__ == '__main__':
    run_bounds()
